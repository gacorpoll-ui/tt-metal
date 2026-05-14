// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <type_traits>
#include <ranges>
#include <optional>
#include <bit>
#include <limits>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor RMSAllGatherMeshWorkloadFactory::create_descriptor(
    const RMSAllGatherParams& operation_attributes,
    const RMSAllGatherInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "RMSAllGatherMeshWorkloadFactory::create_descriptor requires a mesh dispatch coordinate");
    const ttnn::MeshCoordinate mesh_coord = mesh_dispatch_coordinate.value();

    // Setup device information
    ttnn::MeshDevice* mesh_device = tensor_args.input.device();
    auto* const target_device = mesh_device->get_device(mesh_coord);
    const auto mesh_view = mesh_device->get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::vector<IDevice*> devices = (operation_attributes.cluster_axis == 0)
                                        ? mesh_view.get_devices_on_column(mesh_coord[1])
                                        : mesh_view.get_devices_on_row(mesh_coord[0]);
    const auto fabric_node_ids = (operation_attributes.cluster_axis == 0)
                                     ? mesh_view.get_fabric_node_ids_on_column(mesh_coord[1])
                                     : mesh_view.get_fabric_node_ids_on_row(mesh_coord[0]);

    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    uint32_t device_index = 0;
    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (devices.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_fabric_node_id = fabric_node_ids.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_fabric_node_id = fabric_node_ids.at(operation_attributes.ring_size - 1);
            }
            if (i != operation_attributes.ring_size - 1) {
                forward_fabric_node_id = fabric_node_ids.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_fabric_node_id = fabric_node_ids.at(0);
            }
        }
    }

    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& stats = tensor_args.stats;
    auto& output = tensor_return_value;

    // Program creation logic (previously in frmsnorm_multi_core_sharded)
    float eps = operation_attributes.eps;
    uint32_t subblock_wt = operation_attributes.subblock_wt;
    uint32_t block_wt = operation_attributes.block_wt;
    DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;
    uint32_t num_links = operation_attributes.num_links;
    uint32_t ring_size = operation_attributes.ring_size;
    ::ttnn::ccl::Topology topology = operation_attributes.topology;
    const GlobalSemaphore& semaphore = operation_attributes.semaphore;
    bool use_noc1_only = operation_attributes.use_noc1_only;
    uint32_t block_wt_resharded = output.shard_spec().value().shape[1] / TILE_WIDTH;
    bool skip_write_back = output.shard_spec().value() == a.shard_spec().value();

    ////////////////////////////////////////////////////////////////////////////
    //                            Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::ProgramDescriptor desc;
    uint32_t output_page_size = 0;
    uint32_t stats_page_size;
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat stats_data_format = tt::tt_metal::datatype_to_dataformat_converter(stats.value().dtype());
    tt::DataFormat residual_data_format = in_data_format;
    if (b) {
        residual_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.value().dtype());
    }
    if (output.layout() == Layout::TILE) {
        output_page_size = output.tensor_spec().tile().get_tile_size(out_data_format);
    } else {
        output_page_size = output.buffer()->page_size();
    }
    if (stats.value().layout() == Layout::TILE) {
        stats_page_size = stats.value().tensor_spec().tile().get_tile_size(stats_data_format);
    } else {
        stats_page_size = stats.value().buffer()->page_size();
    }

    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;
    if (topology == ccl::Topology::Linear) {
        ccl::LineTopology line_topology(ring_size, device_index);
        num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
        num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
    } else if (topology == ccl::Topology::Ring) {
        num_targets_forward = tt::div_up(ring_size - 1, 2);
        num_targets_backward = ring_size - 1 - num_targets_forward;
        if (device_index % 2 == 0) {
            std::swap(num_targets_forward, num_targets_backward);
        }
    }

    // Get worker cores, assuming 1 worker per link
    ShardSpec shard_spec = a.shard_spec().value();
    CoreRangeSet all_cores = shard_spec.grid;
    const auto& cores = corerange_to_cores(all_cores, all_cores.num_cores(), true);

    // Tensor Info
    const auto input_tensor_cores = a.memory_config().shard_spec()->grid;
    const auto output_tensor_cores = output.memory_config().shard_spec()->grid;
    const auto stats_tensor_cores = stats.value().memory_config().shard_spec()->grid;
    const auto stats_tensor_shard_shape = stats.value().memory_config().shard_spec()->shape;
    const auto stats_tensor_shard_num_pages = stats_tensor_shard_shape[0] * stats_tensor_shard_shape[1] / TILE_HW;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = output_page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;

    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), compute_kernel_config);

    if (!dst_full_sync_en) {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 4,
                "subblock_wt={}, but subblock width must less than 4 tiles in fp32 mode when dst_full_sync_en is false",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles when dst_full_sync_en is false",
                subblock_wt);
        }
    } else {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles in fp32 mode when dst_full_sync_en is true",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 16,
                "subblock_wt={}, but subblock width must less than 16 tiles when dst_full_sync_en is true",
                subblock_wt);
        }
    }
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                              : tt::DataFormat::Float16_b;
    // tile sizes
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t residual_single_tile_size = tt::tile_size(residual_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_data_format);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "res_data_format: {}", residual_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "gamma_cb_data_format: {}", gamma_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t K = shape[-1];
    uint32_t Kt = K / TILE_WIDTH;
    // block
    uint32_t num_blocks = 0;

    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1};
    std::optional<CoreCoord> grid_offset = std::nullopt;
    if (bbox.start_coord.x != 0 || bbox.start_coord.y != 0) {
        grid_offset = bbox.start_coord;
    }
    num_blocks = shard_spec.num_cores();

    // two-stage reduce
    bool use_two_stage_reduce = false;
    // only do this for row/col dim are full length
    if (grid_size.x > 1 && grid_size.x <= mesh_device->compute_with_storage_grid_size().x &&
        grid_size.y > 1) {  // row major and multiple rows
        use_two_stage_reduce = true;
    }
    uint32_t num_subblocks_w = block_wt / subblock_wt;

    // Get all storage cores
    ShardSpec output_shard_spec = output.shard_spec().value();
    CoreRangeSet all_storage_cores = output_shard_spec.grid;
    CoreRangeSet all_worker_and_storage_cores = all_storage_cores.merge(a.shard_spec().value().grid);
    std::vector<uint32_t> storage_core_noc_x;
    std::vector<uint32_t> storage_core_noc_y;
    std::vector<CoreCoord> storage_core_coords =
        corerange_to_cores(all_storage_cores, all_storage_cores.num_cores(), true);
    for (auto core : storage_core_coords) {
        storage_core_noc_x.push_back((std::uint32_t)mesh_device->worker_core_from_logical_core(core).x);
        storage_core_noc_y.push_back((std::uint32_t)mesh_device->worker_core_from_logical_core(core).y);

        log_debug(
            tt::LogOp,
            "Storage core: ({}, {}), physical coords: ({}, {})",
            core.x,
            core.y,
            storage_core_noc_x.back(),
            storage_core_noc_y.back());
    }

    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_block_tiles = block_wt;
    // post_all_gather_stats_block_tiles
    uint32_t post_all_gather_stats_block_tiles = 1;
    uint32_t num_distributed_devices = 1;
    if (stats.has_value()) {
        post_all_gather_stats_block_tiles = stats.value().padded_shape()[-1] / TILE_WIDTH;
        num_distributed_devices = post_all_gather_stats_block_tiles;
    }

    uint32_t in0_CB_size = in0_block_tiles * in_single_tile_size;
    // block size for in1 (tensor b)
    uint32_t in1_CB_size = in0_block_tiles * residual_single_tile_size;
    // in2 - scaler
    uint32_t in2_CB_size = bfloat16_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = bfloat16_tile_size;
    // gamma
    uint32_t in5_CB_size = in0_block_tiles * gamma_single_tile_size / 1;
    // itermediate buffers change later
    uint32_t x_CB_size = in0_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = in0_block_tiles * single_tile_size / block_wt;
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size;
    uint32_t ex_external_CB_size = tt::div_up(Kt, block_wt) * single_tile_size;
    uint32_t stats_cb_size = post_all_gather_stats_block_tiles * single_tile_size;
    uint32_t stats_reduced_cb_size = single_tile_size;
    // output buffer size
    uint32_t out_CB_size;
    out_CB_size = in0_block_tiles * out_single_tile_size;
    uint32_t out_reshard_CB_size = out_CB_size;
    if (!skip_write_back) {
        out_reshard_CB_size = block_wt_resharded * out_single_tile_size;
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    // define core ranges
    bool use_mcast = num_blocks > 1;

    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores_all_to_all = 1;
    uint32_t num_blocks_first_stage = num_blocks;
    uint32_t num_blocks_second_stage = 0;
    if (use_two_stage_reduce) {
        num_blocks_first_stage = num_cores_x;
        num_cores_all_to_all = num_cores_y;
        num_blocks_second_stage = num_cores_y;
    }
    // change tt::CBIndex external size
    if (use_two_stage_reduce) {
        ex_external_CB_size = (num_blocks_first_stage + num_blocks_second_stage - 1) * single_tile_size;
    }
    uint32_t num_none_all_to_all_workers = num_blocks - num_cores_all_to_all;

    CoreCoord start_core = {0, 0};

    CoreRange sender_cores(start_core, start_core);
    CoreRangeSet all_to_all_cores;
    CoreRangeSet all_to_all_workers_except_sender;
    CoreRangeSet not_all_to_all_workers;
    uint32_t num_cores_x_mcast, num_cores_y_mcast;
    sender_cores = {start_core, start_core};
    CoreCoord all_core_grid_size;
    CoreCoord none_core_grid_size;
    if (use_two_stage_reduce) {
        all_core_grid_size = {1, num_cores_y};
        none_core_grid_size = {num_cores_x - 1, num_cores_y};
    } else {
        all_core_grid_size = grid_size;
        none_core_grid_size = grid_size;
    }
    all_to_all_cores = num_cores_to_corerangeset(start_core, num_cores_all_to_all, all_core_grid_size, true);
    if (use_mcast) {
        CoreCoord all_start_core;
        CoreCoord end_core = sender_cores.end_coord;
        if (use_two_stage_reduce) {
            if (end_core.x == all_core_grid_size.x - 1) {
                all_start_core = {0, end_core.y + 1};
            } else {
                all_start_core = {end_core.x + 1, end_core.y};
            }
        } else {
            if (end_core.x == bbox.end_coord.x) {
                all_start_core = {0, end_core.y + 1};
            } else {
                all_start_core = {end_core.x + 1, end_core.y};
            }
        }
        all_to_all_workers_except_sender =
            num_cores_to_corerangeset(all_start_core, num_cores_all_to_all - 1, all_core_grid_size, true);
    }
    if (num_none_all_to_all_workers > 0) {
        if (use_two_stage_reduce) {
            CoreCoord none_start_core = {all_core_grid_size.x, sender_cores.end_coord.y};
            CoreCoord none_end_core = {num_cores_x - 1, num_cores_y - 1};
            CoreRange none_core_range = CoreRange(none_start_core, none_end_core);
            not_all_to_all_workers = CoreRangeSet(none_core_range);
        } else {
            CoreCoord none_start_core;
            CoreCoord end_core = (*all_to_all_cores.ranges().rbegin()).end_coord;
            if (end_core.x == bbox.end_coord.x) {
                none_start_core = {0, end_core.y + 1};
            } else {
                none_start_core = {end_core.x + 1, end_core.y};
            }
            not_all_to_all_workers =
                num_cores_to_corerangeset(none_start_core, num_none_all_to_all_workers, none_core_grid_size, true);
        }
    }
    num_cores_x_mcast = num_cores_x;
    num_cores_y_mcast = num_cores_y;
    auto applyStartOffset = [](const CoreRangeSet& input_set, const CoreCoord& grid_offset) -> CoreRangeSet {
        if (input_set.empty()) {
            return input_set;
        }

        std::vector<CoreRange> new_ranges;
        new_ranges.reserve(input_set.size());

        for (const CoreRange& range : input_set.ranges()) {
            CoreCoord new_start = {range.start_coord.x + grid_offset.x, range.start_coord.y + grid_offset.y};
            CoreCoord new_end = {range.end_coord.x + grid_offset.x, range.end_coord.y + grid_offset.y};
            new_ranges.emplace_back(new_start, new_end);
        }

        return CoreRangeSet(std::move(new_ranges));
    };
    if (grid_offset.has_value()) {
        start_core = {start_core.x + grid_offset.value().x, start_core.y + grid_offset.value().y};
        sender_cores = {
            {sender_cores.start_coord.x + start_core.x, sender_cores.start_coord.y + start_core.y},
            {sender_cores.end_coord.x + start_core.x, sender_cores.end_coord.y + start_core.y}};
        all_to_all_cores = applyStartOffset(all_to_all_cores, grid_offset.value());
        all_to_all_workers_except_sender = applyStartOffset(all_to_all_workers_except_sender, grid_offset.value());
        not_all_to_all_workers = applyStartOffset(not_all_to_all_workers, grid_offset.value());
    }
    // Mcast args.  Each SemaphoreDescriptor is allocated by the framework on cache
    // miss and its assigned id is the index in desc.semaphores.
    const uint32_t reduce_sender_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = reduce_sender_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
    const uint32_t reduce_receiver_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = reduce_receiver_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
    const uint32_t reduce_second_stage_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = reduce_second_stage_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
    const uint32_t post_reduce_sender_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = post_reduce_sender_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
    const uint32_t stats_filled_semaphore = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = stats_filled_semaphore, .core_ranges = all_cores, .initial_value = INVALID});

    // reader defines
    std::map<std::string, std::string> reader_mcast_sender_defines;
    std::map<std::string, std::string> reader_mcast_receiver_defines;
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }

    // compute defines
    // reader defines
    std::map<std::string, std::string> compute_defines;
    if (b.has_value()) {
        compute_defines["FUSE_PRE_ADD"] = "1";
    }

    // Create pre circular buffers

    // in1 sharded

    // in2 scaler
    uint32_t in2_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in2_cb_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = bfloat16_tile_size,
        }}},
    });
    // in4 scaler-c
    uint32_t pre_in4_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(pre_in4_cb_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = bfloat16_tile_size,
        }}},
    });
    // ex_partial2
    uint32_t ex_cb_partial2_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex_partial_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(ex_cb_partial2_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    // ex2
    uint32_t ex2_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(ex2_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // ex_external2
    uint32_t ex_cb_external2_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex_external_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(ex_cb_external2_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // x
    uint32_t pre_x_cb_index = tt::CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = x_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(pre_x_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // out
    uint32_t cb_to_allgather_writer = tt::CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = stats_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_to_allgather_writer),
            .data_format = stats_data_format,
            .page_size = stats_single_tile_size,
        }}},
    });

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_8;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    uint32_t updated_residual_index = tt::CBIndex::c_21;
    uint32_t original_input_index = tt::CBIndex::c_22;
    // All of these take either residual or input as input, one for pre, one for post
    uint32_t in0_cb_index = tt::CBIndex::c_12;
    uint32_t pre_in0_cb_index = tt::CBIndex::c_5;

    if (b) {
        // Tensors that do the b fusing.  cb_in1 is bound to input tensor a,
        // cb_add_out/cb_in0/pre_cb_in0 share residual tensor b's buffer.  Setting
        // CBDescriptor::buffer tells the framework to auto-patch the dynamic
        // base address on every dispatch (replacing UpdateDynamicCircularBufferAddress).
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(original_input_index),
                .data_format = in_data_format,
                .page_size = in_single_tile_size,
            }}},
            .buffer = a.buffer(),
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(updated_residual_index),
                .data_format = residual_data_format,
                .page_size = residual_single_tile_size,
            }}},
            .buffer = b.value().buffer(),
        });

        // Other CBs should use the now updated b as an input
        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in0_cb_index),
                .data_format = residual_data_format,
                .page_size = residual_single_tile_size,
            }}},
            .buffer = b.value().buffer(),
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(pre_in0_cb_index),
                .data_format = residual_data_format,
                .page_size = residual_single_tile_size,
            }}},
            .buffer = b.value().buffer(),
        });
    } else {
        // There is no b so just use a as the input
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in0_cb_index),
                .data_format = in_data_format,
                .page_size = in_single_tile_size,
            }}},
            .buffer = a.buffer(),
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(pre_in0_cb_index),
                .data_format = in_data_format,
                .page_size = in_single_tile_size,
            }}},
            .buffer = a.buffer(),
        });
    }
    // Create post circular buffers

    // ex_global
    uint32_t ex_global_cb_index = tt::CBIndex::c_9;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex_global_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(ex_global_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // out
    uint32_t output_cb_index = tt::CBIndex::c_10;
    {
        CBDescriptor output_cb_desc{
            .total_size = out_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = out_data_format,
                .page_size = out_single_tile_size,
            }}},
        };
        if (skip_write_back) {
            output_cb_desc.buffer = output.buffer();
        }
        desc.cbs.push_back(std::move(output_cb_desc));
    }

    // gamma
    uint32_t in5_cb_index = tt::CBIndex::c_11;
    if (gamma.has_value()) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in5_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in5_cb_index),
                .data_format = gamma_cb_data_format,
                .page_size = gamma_single_tile_size,
            }}},
        });
    }

    // in3 eps
    uint32_t in3_cb_index = tt::CBIndex::c_13;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in3_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in3_cb_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = bfloat16_tile_size,
        }}},
    });

    // in4 scaler-c
    uint32_t in4_cb_index = tt::CBIndex::c_14;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in4_cb_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = bfloat16_tile_size,
        }}},
    });

    // x
    uint32_t x_cb_index;
    x_cb_index = tt::CBIndex::c_15;
    desc.cbs.push_back(CBDescriptor{
        .total_size = x_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(x_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    uint32_t output_reshard_cb_index = tt::CBIndex::c_16;
    if (!skip_write_back) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = out_reshard_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_reshard_cb_index),
                .data_format = out_data_format,
                .page_size = out_single_tile_size,
            }}},
            .buffer = output.buffer(),
        });
    }

    // cb_var
    uint32_t cb_var_index = tt::CBIndex::c_17;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex_global_CB_size,
        .core_ranges = sender_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_var_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // cb_stats_reduced
    uint32_t cb_stats_reduced_index;
    cb_stats_reduced_index = tt::CBIndex::c_18;
    desc.cbs.push_back(CBDescriptor{
        .total_size = stats_reduced_cb_size,
        .core_ranges = sender_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_stats_reduced_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // cb_stats
    uint32_t cb_stats_index = tt::CBIndex::c_19;
    desc.cbs.push_back(CBDescriptor{
        .total_size = stats_cb_size,
        .core_ranges = sender_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_stats_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = stats.value().buffer(),
    });
    uint32_t signaling_cb = tt::CBIndex::c_20;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(signaling_cb),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2,
        }}},
    });

    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)num_cores_x_mcast,
        (std::uint32_t)num_cores_y_mcast,
        (std::uint32_t)use_two_stage_reduce,
        (std::uint32_t)num_blocks_first_stage,
        (std::uint32_t)num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)ex_cb_partial2_index,
        (std::uint32_t)ex2_cb_index,
        (std::uint32_t)ex_cb_external2_index,
        (std::uint32_t)post_reduce_sender_semaphore_id,
        (std::uint32_t)cb_stats_reduced_index,
        (std::uint32_t)ex_global_cb_index};
    std::vector<uint32_t> reader_mcast_receiver_all_to_all_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)1,
        (std::uint32_t)num_cores_x_mcast,
        (std::uint32_t)num_cores_y_mcast,
        (std::uint32_t)use_two_stage_reduce,
        (std::uint32_t)num_blocks_first_stage,
        (std::uint32_t)num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)ex_cb_partial2_index,
        (std::uint32_t)ex2_cb_index,
        (std::uint32_t)ex_cb_external2_index,
        (std::uint32_t)post_reduce_sender_semaphore_id,
        (std::uint32_t)ex_global_cb_index};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)0,
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)ex_cb_partial2_index,
        (std::uint32_t)ex2_cb_index,
        (std::uint32_t)ex_cb_external2_index,
        (std::uint32_t)post_reduce_sender_semaphore_id,
        (std::uint32_t)ex_global_cb_index};

    std::vector<uint32_t> writer_compile_time_args = {
        1,  // Gets overwritten in not all to all workers
        in2_cb_index,
        pre_in4_cb_index,
        cb_to_allgather_writer,
        // all gather parameters
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_pages_per_packet,             // packet_size_in_pages
        stats_page_size,                  // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        num_links,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)block_wt,
        output_reshard_cb_index,
        output_cb_index,
        in3_cb_index,
        in4_cb_index,
        in5_cb_index};
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        writer_compile_time_args.push_back(gamma_stick_size);
    } else {
        writer_compile_time_args.push_back(0);
    }

    writer_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);

    // write back compile time args
    writer_compile_time_args.push_back(block_wt * out_single_tile_size);  // out_tensor_stride_w_bytes
    writer_compile_time_args.push_back(
        block_wt_resharded * out_single_tile_size);  // out_reshard_tensor_stride_w_bytes: how many bytes to skip to get
                                                     // to the next data chunk
    writer_compile_time_args.push_back(stats_filled_semaphore);
    writer_compile_time_args.push_back(signaling_cb);
    writer_compile_time_args.push_back(num_blocks);
    tt::tt_metal::TensorAccessorArgs(gamma ? gamma->buffer() : nullptr).append_to(writer_compile_time_args);

    tt::tt_metal::NOC reader_noc = NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = NOC::NOC_1;
    if (!use_noc1_only) {
        reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch());
        writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch());

        if (!skip_write_back) {
            reader_noc = NOC::NOC_0;
            writer_noc = NOC::NOC_1;
        }
    }

    // compute kernel compile time args
    std::vector<uint32_t> compute_compile_time_args = {
        num_blocks_first_stage,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        1,  // To be overwritten if not an all to all worker
        block_wt,
        fp32_dest_acc_en,
        num_blocks_second_stage,
        in2_cb_index,
        pre_in4_cb_index,
        ex_cb_partial2_index,
        ex2_cb_index,
        updated_residual_index,
        ex_cb_external2_index,
        cb_to_allgather_writer,
        pre_x_cb_index,
        original_input_index,
        pre_in0_cb_index,
        output_cb_index,
        cb_stats_index,
        in0_cb_index,
        in3_cb_index,
        in4_cb_index,
        cb_var_index,
        x_cb_index,
        in5_cb_index,
        cb_stats_reduced_index,
        ex_global_cb_index,
        signaling_cb};

    // reader kernel

    std::string sender_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_sender_reader.cpp";
    std::string receiver_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp";

    std::map<std::string, std::string> writer_defines;
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }

    // Helper to convert std::map defines to KernelDescriptor::Defines vector form.
    auto to_defines_vec = [](const std::map<std::string, std::string>& m) {
        KernelDescriptor::Defines out;
        out.reserve(m.size());
        for (const auto& kv : m) {
            out.emplace_back(kv.first, kv.second);
        }
        return out;
    };

    auto make_dm_config = [&](tt::tt_metal::DataMovementProcessor processor, tt::tt_metal::NOC noc) {
        return DataMovementConfigDescriptor{
            .processor = processor,
            .noc = noc,
            .noc_mode =
                (use_noc1_only) ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
        };
    };

    // Build kernel descriptors.  Push them onto desc.kernels NOW (before the
    // per-core runtime-args loop) so we can refer to them by stable index for
    // both emplace_runtime_args() and the fabric helper, which expects a
    // KernelHandle that indexes into desc.kernels for the ProgramDescriptor
    // overload.
    KernelDescriptor reader_mcast_sender_kernel_desc;
    reader_mcast_sender_kernel_desc.kernel_source = sender_reader_kernel_file;
    reader_mcast_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_mcast_sender_kernel_desc.core_ranges = sender_cores;
    reader_mcast_sender_kernel_desc.compile_time_args = std::move(reader_mcast_sender_compile_time_args);
    reader_mcast_sender_kernel_desc.defines = to_defines_vec(reader_mcast_sender_defines);
    reader_mcast_sender_kernel_desc.config = make_dm_config(DataMovementProcessor::RISCV_0, reader_noc);
    const tt::tt_metal::KernelHandle reader_mcast_sender_kernels_id = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(reader_mcast_sender_kernel_desc));

    tt::tt_metal::KernelHandle reader_mcast_receiver_kernels_id_all_to_all = std::numeric_limits<KernelHandle>::max();
    tt::tt_metal::KernelHandle reader_mcast_receiver_kernels_id = std::numeric_limits<KernelHandle>::max();
    if (use_mcast) {
        KernelDescriptor receiver_kernel_desc;
        receiver_kernel_desc.kernel_source = receiver_reader_kernel_file;
        receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        receiver_kernel_desc.core_ranges = all_to_all_workers_except_sender;
        receiver_kernel_desc.compile_time_args = reader_mcast_receiver_all_to_all_compile_time_args;
        receiver_kernel_desc.defines = to_defines_vec(reader_mcast_receiver_defines);
        receiver_kernel_desc.config = make_dm_config(DataMovementProcessor::RISCV_0, reader_noc);
        reader_mcast_receiver_kernels_id_all_to_all = static_cast<KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(receiver_kernel_desc));
    }
    if (num_none_all_to_all_workers > 0) {
        KernelDescriptor receiver_kernel_desc;
        receiver_kernel_desc.kernel_source = receiver_reader_kernel_file;
        receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        receiver_kernel_desc.core_ranges = not_all_to_all_workers;
        receiver_kernel_desc.compile_time_args = std::move(reader_mcast_receiver_compile_time_args);
        receiver_kernel_desc.defines = to_defines_vec(reader_mcast_receiver_defines);
        receiver_kernel_desc.config = make_dm_config(DataMovementProcessor::RISCV_0, reader_noc);
        reader_mcast_receiver_kernels_id = static_cast<KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(receiver_kernel_desc));
    }

    // writer kernel + all gather kernel
    std::string writer_kernel =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp";
    KernelDescriptor writer_sender_kernel_desc;
    writer_sender_kernel_desc.kernel_source = writer_kernel;
    writer_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_sender_kernel_desc.core_ranges = all_to_all_cores;
    writer_sender_kernel_desc.compile_time_args = writer_compile_time_args;  // copy; writer_compile_time_args[0]
                                                                             // is patched below for the receiver
                                                                             // variant.
    writer_sender_kernel_desc.defines = to_defines_vec(writer_defines);
    writer_sender_kernel_desc.config = make_dm_config(DataMovementProcessor::RISCV_1, writer_noc);
    const tt::tt_metal::KernelHandle writer_mcast_sender_kernels_id = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_sender_kernel_desc));

    tt::tt_metal::KernelHandle writer_mcast_receiver_kernels_id = std::numeric_limits<KernelHandle>::max();
    if (num_none_all_to_all_workers > 0) {
        writer_compile_time_args.at(0) = 0;
        KernelDescriptor writer_receiver_kernel_desc;
        writer_receiver_kernel_desc.kernel_source = writer_kernel;
        writer_receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_receiver_kernel_desc.core_ranges = not_all_to_all_workers;
        writer_receiver_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
        writer_receiver_kernel_desc.defines = to_defines_vec(writer_defines);
        writer_receiver_kernel_desc.config = make_dm_config(DataMovementProcessor::RISCV_1, writer_noc);
        writer_mcast_receiver_kernels_id = static_cast<KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(writer_receiver_kernel_desc));
    }

    // compute kernel
    std::string compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/compute/"
        "rms_compute.cpp";
    KernelDescriptor compute_kernel_all_to_all_desc;
    compute_kernel_all_to_all_desc.kernel_source = compute_kernel_file;
    compute_kernel_all_to_all_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_all_to_all_desc.core_ranges = all_to_all_cores;
    compute_kernel_all_to_all_desc.compile_time_args = compute_compile_time_args;  // copy
    compute_kernel_all_to_all_desc.defines = to_defines_vec(compute_defines);
    compute_kernel_all_to_all_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };
    const tt::tt_metal::KernelHandle compute_kernels_id_all_to_all = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(compute_kernel_all_to_all_desc));

    tt::tt_metal::KernelHandle compute_kernels_id = std::numeric_limits<KernelHandle>::max();
    if (num_none_all_to_all_workers > 0) {
        compute_compile_time_args.at(4) = 0;
        KernelDescriptor compute_kernel_desc;
        compute_kernel_desc.kernel_source = compute_kernel_file;
        compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_kernel_desc.core_ranges = not_all_to_all_workers;
        compute_kernel_desc.compile_time_args = std::move(compute_compile_time_args);
        compute_kernel_desc.defines = to_defines_vec(compute_defines);
        compute_kernel_desc.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
        };
        compute_kernels_id = static_cast<KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(compute_kernel_desc));
    }

    // Runtime Args
    float cinv_pre = (1.0f / num_blocks);  // bcast-cores scaler
    float cinv = (1.0f / num_distributed_devices);
    float cinv_one = 1.0f;  // bcast-cores scaler for all-to-all cores not on first row/col
    uint32_t cinv_bits = std::bit_cast<uint32_t>(cinv);
    uint32_t cinv_pre_bits = std::bit_cast<uint32_t>(cinv_pre);
    uint32_t cinv_one_bits = std::bit_cast<uint32_t>(cinv_one);
    uint32_t eps_bits = std::bit_cast<uint32_t>(eps);

    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    in0_mcast_noc_x.reserve(num_cores_x);
    in0_mcast_noc_y.reserve(num_cores_y);
    CoreCoord core_start_offset = grid_offset.value_or(CoreCoord{0, 0});
    for (uint32_t core_idx_x = core_start_offset.x; core_idx_x < num_cores_x + core_start_offset.x; ++core_idx_x) {
        in0_mcast_noc_x.push_back(mesh_device->worker_core_from_logical_core({core_idx_x, core_start_offset.y}).x);
    }
    for (uint32_t core_idx_y = core_start_offset.y; core_idx_y < num_cores_y + core_start_offset.y; ++core_idx_y) {
        in0_mcast_noc_y.push_back(mesh_device->worker_core_from_logical_core({core_start_offset.x, core_idx_y}).y);
    }

    uint32_t last_core_width_index = 0;
    last_core_width_index = cores.size() - 1;

    // For write back calculation
    uint32_t current_storage_core = 0;
    uint32_t current_storage_core_offset = 0;

    // All gather RT prep

    CoreCoord drain_sync_core;
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto stats_cores_vec = corerange_to_cores(stats_tensor_cores, std::nullopt, true);
    auto cores_per_device = stats_cores_vec.size() + ring_size - (1 / ring_size);
    uint32_t start_core_index_for_device = stats_cores_vec.size() / ring_size * device_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;
    auto stats_cores_this_device = std::vector<CoreCoord>(
        stats_cores_vec.begin() + start_core_index_for_device, stats_cores_vec.begin() + end_core_index_for_device);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        log_debug(tt::LogOp, "core: {}, {}", core.x, core.y);

        uint32_t width_index = 0;
        width_index = i;

        uint32_t width_index_two_stage = width_index % num_blocks_first_stage;

        uint32_t all_to_all_worker_tile_offset_size_bytes;
        if (use_two_stage_reduce) {
            all_to_all_worker_tile_offset_size_bytes = (width_index_two_stage)*single_tile_size;
        } else {
            all_to_all_worker_tile_offset_size_bytes = (width_index)*single_tile_size;
        }
        uint32_t gamma_tile_start_id = width_index * block_wt;
        uint32_t num_reduce_tiles_per_block_h = block_wt;
        // account for padding
        if (width_index == last_core_width_index) {
            num_reduce_tiles_per_block_h = Kt - last_core_width_index * block_wt;
        }

        std::vector<uint32_t> compute_args{num_reduce_tiles_per_block_h};
        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            compute_args.push_back(1);
            compute_args.push_back((uint32_t)use_two_stage_reduce);
            bool is_second_stage_reader;
            if (use_two_stage_reduce) {
                is_second_stage_reader = width_index < 1;
            } else {
                is_second_stage_reader = false;
            }
            compute_args.push_back((uint32_t)is_second_stage_reader);
            compute_args.push_back((uint32_t)(!(use_two_stage_reduce && (!is_second_stage_reader))));
            compute_args.push_back((uint32_t)num_distributed_devices);
            desc.kernels[compute_kernels_id_all_to_all].runtime_args.emplace_back(core, std::move(compute_args));
        } else {
            desc.kernels[compute_kernels_id].runtime_args.emplace_back(core, std::move(compute_args));
        }

        if (width_index == 0) {
            CoreCoord mcast_start, mcast_end;
            CoreCoord top_left_core = {(std::size_t)start_core.x, (std::size_t)start_core.y};
            CoreCoord bottom_right_core = {
                (std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1};
            auto top_left_core_physical = mesh_device->worker_core_from_logical_core(top_left_core);
            auto bottom_right_core_physical = mesh_device->worker_core_from_logical_core(bottom_right_core);
            mcast_start = top_left_core_physical;
            mcast_end = bottom_right_core_physical;
            if (reader_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }
            std::vector<uint32_t> mcast_sender_args;
            mcast_sender_args.push_back(mcast_start.x);
            mcast_sender_args.push_back(mcast_start.y);
            mcast_sender_args.push_back(mcast_end.x);
            mcast_sender_args.push_back(mcast_end.y);
            mcast_sender_args.push_back(core.x - start_core.x);
            mcast_sender_args.push_back(core.y - start_core.y);
            mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
            mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            desc.kernels[reader_mcast_sender_kernels_id].runtime_args.emplace_back(core, std::move(mcast_sender_args));
        } else if (
            (not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            std::vector<uint32_t> mcast_receiver_args;
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            bool is_second_stage_reader;
            if (use_two_stage_reduce and width_index < 1) {
                is_second_stage_reader = true;
                mcast_receiver_args.push_back((uint32_t)is_second_stage_reader);
            } else {
                is_second_stage_reader = false;
                mcast_receiver_args.push_back((uint32_t)is_second_stage_reader);
            }
            mcast_receiver_args.push_back(core.x - start_core.x);
            mcast_receiver_args.push_back(core.y - start_core.y);
            mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
            mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            desc.kernels[reader_mcast_receiver_kernels_id_all_to_all].runtime_args.emplace_back(
                core, std::move(mcast_receiver_args));
        } else {
            std::vector<uint32_t> mcast_receiver_args;
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(in0_mcast_noc_x[0]);
            mcast_receiver_args.push_back(in0_mcast_noc_y[0]);
            desc.kernels[reader_mcast_receiver_kernels_id].runtime_args.emplace_back(
                core, std::move(mcast_receiver_args));
        }
        // Set all gather runtime args

        uint32_t out_ready_sem_wait_value = ring_size * num_links;
        // all_gather_rts Start at RT index 3 of writer
        std::vector<uint32_t> all_gather_rts = {
            semaphore.address(),               // out_ready_sem_bank_addr (absolute address)
            out_ready_sem_wait_value,          // out_ready_sem_wait_value
            stats.value().buffer()->address()  // tensor_address0
        };

        if (i == 0) {
            // Add RT values for the all gather core to all_gather_rts
            // Will be appended to the end of writer rt args
            uint32_t base_pages_per_worker = 1 / num_links;
            uint32_t remainder = 1 % num_links;
            uint32_t input_tile_id_start = (i * base_pages_per_worker) + std::min(i, remainder);
            uint32_t input_tile_id_end = ((i + 1) * base_pages_per_worker) + std::min(i + 1, remainder);
            uint32_t stats_first_core_tile_start_offset =
                (device_index + input_tile_id_start) % stats_tensor_shard_num_pages;
            std::vector<uint32_t> stats_tensor_cores_x;
            std::vector<uint32_t> stats_tensor_cores_y;
            for (uint32_t i = input_tile_id_start / stats_tensor_shard_num_pages;
                 i < (input_tile_id_end + stats_tensor_shard_num_pages - 1) / stats_tensor_shard_num_pages;
                 i++) {
                auto this_core = mesh_device->worker_core_from_logical_core(stats_cores_this_device[i]);
                stats_tensor_cores_x.push_back(this_core.x);
                stats_tensor_cores_y.push_back(this_core.y);
            }
            if (i == 0) {
                // drain sync core is the first worker core
                drain_sync_core = mesh_device->worker_core_from_logical_core(core);
            }

            std::vector<uint32_t> base_rt_args = {
                stats_first_core_tile_start_offset,  // first_core_tile_start_offset
                stats_tensor_cores_x.size(),         // num_cores
                drain_sync_core.x,                   // out_ready_sem_noc0_x
                drain_sync_core.y                    // out_ready_sem_noc0_y
            };
            all_gather_rts.insert(all_gather_rts.end(), base_rt_args.begin(), base_rt_args.end());
            all_gather_rts.insert(all_gather_rts.end(), stats_tensor_cores_x.begin(), stats_tensor_cores_x.end());
            all_gather_rts.insert(all_gather_rts.end(), stats_tensor_cores_y.begin(), stats_tensor_cores_y.end());

            all_gather_rts.push_back(forward_fabric_node_id.has_value());
            if (forward_fabric_node_id.has_value()) {
                const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    target_device_fabric_node_id, forward_fabric_node_id.value(), i, desc, core, all_gather_rts);
            }

            all_gather_rts.push_back(backward_fabric_node_id.has_value());
            if (backward_fabric_node_id.has_value()) {
                const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    target_device_fabric_node_id, backward_fabric_node_id.value(), i, desc, core, all_gather_rts);
            }
        }
        // Set writer runtime args
        std::vector<uint32_t> write_back_writer_args;

        uint32_t num_storage_cores = all_storage_cores.num_cores();

        write_back_writer_args.push_back(
            current_storage_core_offset * out_single_tile_size);  // storage_core_start_offset

        uint32_t current_worker_num_segments_to_write_back = 0;
        uint32_t worker_core_current_offset = 0;

        while (worker_core_current_offset <
               block_wt) {  // Continue until all worker core data has been written to corresponding storage cores
            uint32_t num_tiles_available_at_current_storage_core = block_wt_resharded - current_storage_core_offset;
            uint32_t num_tiles_left_on_current_worker_core = block_wt - worker_core_current_offset;
            uint32_t num_tiles_to_write_back =
                std::min(num_tiles_left_on_current_worker_core, num_tiles_available_at_current_storage_core);
            current_worker_num_segments_to_write_back += 1;

            log_debug(
                tt::LogOp,
                "New segment for worker core {}, Worker core offset: {}, Storage core offset: {}, Num tiles to "
                "write "
                "back: {}",
                i,
                worker_core_current_offset,
                current_storage_core_offset,
                num_tiles_to_write_back);

            write_back_writer_args.push_back(num_tiles_to_write_back);                   // num_bytes_to_write_back
            write_back_writer_args.push_back(storage_core_noc_x[current_storage_core]);  // current_storage_core_noc_x
            write_back_writer_args.push_back(storage_core_noc_y[current_storage_core]);  // current_storage_core_noc_y
            worker_core_current_offset += num_tiles_to_write_back;
            current_storage_core_offset += num_tiles_to_write_back;

            if (current_storage_core_offset >= block_wt_resharded) {
                current_storage_core += 1;        // Move to next storage core
                current_storage_core_offset = 0;  // Reset offset on new storage core

                TT_ASSERT(
                    current_storage_core <= num_storage_cores,
                    "current_storage_core {} is exceeding number of storage cores {}",
                    current_storage_core,
                    num_storage_cores);
            }
        }
        TT_ASSERT(
            worker_core_current_offset == block_wt,
            "All worker core data should be written, but worker_core_current_offset {} != block_wt {}",
            worker_core_current_offset,
            block_wt);

        write_back_writer_args.insert(write_back_writer_args.begin(), current_worker_num_segments_to_write_back);

        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            std::vector<uint32_t> writer_mcast_sender_args = {0};
            CoreCoord mcast_start, mcast_end;
            CoreCoord top_left_core = {(std::size_t)start_core.x, (std::size_t)start_core.y};
            CoreCoord bottom_right_core = {
                (std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1};
            auto top_left_core_physical = mesh_device->worker_core_from_logical_core(top_left_core);
            auto bottom_right_core_physical = mesh_device->worker_core_from_logical_core(bottom_right_core);
            mcast_start = top_left_core_physical;
            mcast_end = bottom_right_core_physical;
            if (writer_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }
            writer_mcast_sender_args.push_back(mcast_start.x);
            writer_mcast_sender_args.push_back(mcast_start.y);
            writer_mcast_sender_args.push_back(mcast_end.x);
            writer_mcast_sender_args.push_back(mcast_end.y);
            if (use_two_stage_reduce && (!(width_index < 1))) {
                writer_mcast_sender_args.push_back(cinv_one_bits);
            } else {
                writer_mcast_sender_args.push_back(cinv_pre_bits);
            }
            writer_mcast_sender_args.push_back(i);  // Core ID to limit number of cores to do all gather on
            writer_mcast_sender_args.insert(
                writer_mcast_sender_args.end(), all_gather_rts.begin(), all_gather_rts.end());
            writer_mcast_sender_args.at(0) = writer_mcast_sender_args.size();
            std::vector<uint32_t> writer_mcast_post_sender_args;
            if (use_two_stage_reduce) {
                if (width_index < 1) {
                    writer_mcast_post_sender_args.push_back(cinv_bits);
                } else {
                    writer_mcast_post_sender_args.push_back(cinv_one_bits);
                }
            } else {
                writer_mcast_post_sender_args.push_back(cinv_bits);
            }
            writer_mcast_post_sender_args.push_back(eps_bits);
            writer_mcast_post_sender_args.push_back(gamma_dram_addr);
            writer_mcast_post_sender_args.push_back(gamma_tile_start_id);

            // Add args for write back (reshard)
            writer_mcast_post_sender_args.insert(
                writer_mcast_post_sender_args.end(), write_back_writer_args.begin(), write_back_writer_args.end());
            writer_mcast_sender_args.insert(
                writer_mcast_sender_args.end(),
                writer_mcast_post_sender_args.begin(),
                writer_mcast_post_sender_args.end());
            desc.kernels[writer_mcast_sender_kernels_id].runtime_args.emplace_back(
                core, std::move(writer_mcast_sender_args));
        } else {
            std::vector<uint32_t> writer_mcast_receiver_args = {0};
            CoreCoord mcast_start, mcast_end;
            CoreCoord top_left_core = {(std::size_t)start_core.x, (std::size_t)start_core.y};
            CoreCoord bottom_right_core = {
                (std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1};
            auto top_left_core_physical = mesh_device->worker_core_from_logical_core(top_left_core);
            auto bottom_right_core_physical = mesh_device->worker_core_from_logical_core(bottom_right_core);
            mcast_start = top_left_core_physical;
            mcast_end = bottom_right_core_physical;
            if (reader_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }
            writer_mcast_receiver_args.push_back(mcast_start.x);
            writer_mcast_receiver_args.push_back(mcast_start.y);
            writer_mcast_receiver_args.push_back(mcast_end.x);
            writer_mcast_receiver_args.push_back(mcast_end.y);
            writer_mcast_receiver_args.push_back(cinv_pre_bits);
            writer_mcast_receiver_args.push_back(i);  // Core ID to limit number of cores to do all gather on
            writer_mcast_receiver_args.insert(
                writer_mcast_receiver_args.end(), all_gather_rts.begin(), all_gather_rts.end());
            writer_mcast_receiver_args.at(0) = writer_mcast_receiver_args.size();
            std::vector<uint32_t> writer_mcast_post_receiver_args;
            writer_mcast_post_receiver_args.push_back(cinv_bits);
            writer_mcast_post_receiver_args.push_back(eps_bits);
            writer_mcast_post_receiver_args.push_back(gamma_dram_addr);
            writer_mcast_post_receiver_args.push_back(gamma_tile_start_id);
            // Add args for write back (reshard)
            writer_mcast_post_receiver_args.insert(
                writer_mcast_post_receiver_args.end(), write_back_writer_args.begin(), write_back_writer_args.end());
            writer_mcast_receiver_args.insert(
                writer_mcast_receiver_args.end(),
                writer_mcast_post_receiver_args.begin(),
                writer_mcast_post_receiver_args.end());
            desc.kernels[writer_mcast_receiver_kernels_id].runtime_args.emplace_back(
                core, std::move(writer_mcast_receiver_args));
        }
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
