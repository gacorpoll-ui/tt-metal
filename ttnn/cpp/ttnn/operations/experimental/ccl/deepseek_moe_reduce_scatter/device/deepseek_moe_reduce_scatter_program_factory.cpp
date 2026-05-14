// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_program_factory.hpp"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/math.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

CoreCoord choose_additional_core(
    ttnn::MeshDevice* mesh_device, const std::vector<CoreCoord>& cores_already_selected, uint32_t clamped_num_links) {
    /*
     * - optimal core to use as the additional core (when necessary), so that each used link has both a forward and
     * backward worker
     * - respective core is only optimal when the optimal shard grid is used for the input tensors
     */
    constexpr std::array optimal_supplemental_core_per_link = {
        CoreCoord(2, 5),
        CoreCoord(3, 5),
        CoreCoord(6, 5),
        CoreCoord(0, 5),
    };

    // try optimal core first
    CoreCoord optimal_supplemental_core = optimal_supplemental_core_per_link.at(clamped_num_links - 1);
    if (std::find(cores_already_selected.begin(), cores_already_selected.end(), optimal_supplemental_core) ==
        cores_already_selected.end()) {
        return optimal_supplemental_core;
    }

    // try to find any other available core
    auto available_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0));
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                CoreCoord core = CoreCoord(x, y);
                if (std::find(cores_already_selected.begin(), cores_already_selected.end(), core) ==
                    cores_already_selected.end()) {
                    return core;
                }
            }
        }
    }

    TT_FATAL(false, "deepseek_moe_reduce_scatter requires an even number of worker cores");
}

std::tuple<uint32_t, CoreRangeSet, std::vector<CoreCoord>> get_cores(
    ttnn::MeshDevice* mesh_device,
    const NdShardSpec& input_nd_shard_spec,
    uint32_t num_shards,
    uint32_t num_directions_per_link) {
    uint32_t clamped_num_links = tt::div_up(num_shards, num_directions_per_link);

    std::vector<CoreCoord> worker_cores = corerange_to_cores(
        input_nd_shard_spec.grid, num_shards, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    TT_FATAL(
        worker_cores.size() == num_shards,
        "deepseek_moe_reduce_scatter requires each shard to be located on a different core");

    // always need a forward and backward core for each link being used (for in op synchronization), even if the forward
    // worker isn't being used for data transfer due to an odd number of shards
    if (num_shards % 2 != 0) {
        worker_cores.emplace_back(choose_additional_core(mesh_device, worker_cores, clamped_num_links));
    }

    std::vector<CoreRange> worker_core_ranges;
    worker_core_ranges.reserve(worker_cores.size());
    for (const CoreCoord& worker_core : worker_cores) {
        worker_core_ranges.emplace_back(worker_core);
    }
    CoreRangeSet worker_core_range_set = CoreRangeSet(worker_core_ranges);

    return {clamped_num_links, worker_core_range_set, worker_cores};
}

}  // namespace

namespace ttnn::experimental::prim {

DeepseekMoEReduceScatterMeshWorkloadFactory::Resources DeepseekMoEReduceScatterMeshWorkloadFactory::prepare_resources(
    const DeepseekMoEReduceScatterParams& /*operation_attributes*/,
    const DeepseekMoEReduceScatterInputs& tensor_args,
    std::vector<ttnn::Tensor>& /*tensor_return_value*/) {
    auto* mesh_device = tensor_args.input_tensors.at(0).device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // 1 semaphore used for within op synchronizations
    tt::tt_metal::GlobalSemaphore op_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);

    // 1 semaphore used for pre op synchronization to ensure intermediate/output tensors are allocated
    tt::tt_metal::GlobalSemaphore pre_op_semaphore_barrier =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);

    ttnn::SmallVector<tt::tt_metal::SubDeviceId> sub_device_ids = {sd_id};
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, sub_device_ids);

    Resources resources;
    resources.op_semaphore = std::move(op_semaphore);
    resources.pre_op_barrier_semaphore = std::move(pre_op_semaphore_barrier);
    return resources;
}

tt::tt_metal::ProgramDescriptor DeepseekMoEReduceScatterMeshWorkloadFactory::create_descriptor(
    const DeepseekMoEReduceScatterParams& operation_attributes,
    const DeepseekMoEReduceScatterInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    Resources& workload_resources,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "DeepseekMoEReduceScatterMeshWorkloadFactory::create_descriptor requires a mesh dispatch coordinate");
    const ttnn::MeshCoordinate mesh_coordinate = mesh_dispatch_coordinate.value();
    const tt::tt_metal::GlobalSemaphore& op_semaphore = workload_resources.op_semaphore.value();
    const tt::tt_metal::GlobalSemaphore& pre_op_barrier_semaphore = workload_resources.pre_op_barrier_semaphore.value();

    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    const std::vector<ttnn::Tensor> intermediate_slice_tensors(
        tensor_return_value.begin(), tensor_return_value.end() - 1);  // first 8 are intermediate tensors
    const ttnn::Tensor& output_tensor = tensor_return_value.back();   // last is the output tensor

    std::optional<uint32_t> cluster_axis = operation_attributes.cluster_axis;

    const std::optional<ttnn::MeshCoordinate> forward_coordinate =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensors.at(0), mesh_coordinate, 1, tt::tt_fabric::Topology::Ring, cluster_axis);
    const std::optional<ttnn::MeshCoordinate> backward_coordinate =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensors.at(0), mesh_coordinate, -1, tt::tt_fabric::Topology::Ring, cluster_axis);
    TT_FATAL(
        forward_coordinate.has_value() && backward_coordinate.has_value(),
        "DEBUG: forward_coord or backward_coord is null");

    uint32_t ring_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensors.at(0), mesh_coordinate, cluster_axis);
    log_debug(tt::LogOp, "Device index for {} is {}", mesh_coordinate, ring_index);

    auto* mesh_device = input_tensors.at(0).device();
    const uint32_t num_links = operation_attributes.num_links;

    // hardcoded constants
    const uint32_t ring_size = 8;
    const uint32_t num_directions_per_link = 2;
    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    // tensor details
    const NdShardSpec& input_nd_shard_spec = input_tensors.at(0).nd_shard_spec().value();
    const uint32_t num_pages_per_shard = input_nd_shard_spec.shard_shape.volume() / num_tile_elements;
    const uint32_t num_shards = input_tensors.at(0).physical_volume() / (num_tile_elements * num_pages_per_shard);
    const uint32_t num_pages_per_slice = input_tensors.at(0).buffer()->num_pages();
    const uint32_t page_size = input_tensors.at(0).buffer()->page_size();

    // choose cores
    const auto [clamped_num_links, worker_core_range_set, worker_cores] =
        get_cores(mesh_device, input_nd_shard_spec, num_shards, num_directions_per_link);
    TT_FATAL(clamped_num_links <= num_links, "{} links available, but {} requested", num_links, clamped_num_links);

    // NOTE: writer kernel hardcoded to always use scatter_write with 2 tiles
    const uint32_t tile_granularity = 2;

    // L1 scratch CB creation
    const uint32_t compute_input_cb_num_pages = num_pages_per_shard;   // entire shard
    const uint32_t compute_ouput_cb_num_pages = 2 * tile_granularity;  // double buffer

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype());

    uint32_t input_slice_0_cb_id = tt::CBIndex::c_0;
    uint32_t input_slice_1_cb_id = tt::CBIndex::c_1;
    uint32_t input_slice_2_cb_id = tt::CBIndex::c_2;
    uint32_t input_slice_3_cb_id = tt::CBIndex::c_3;
    uint32_t input_slice_4_cb_id = tt::CBIndex::c_4;
    uint32_t input_slice_5_cb_id = tt::CBIndex::c_5;
    uint32_t input_slice_6_cb_id = tt::CBIndex::c_6;
    uint32_t input_slice_7_cb_id = tt::CBIndex::c_7;

    uint32_t intermediate_slice_0_cb_id = tt::CBIndex::c_8;
    uint32_t intermediate_slice_1_cb_id = tt::CBIndex::c_9;
    uint32_t intermediate_slice_2_cb_id = tt::CBIndex::c_10;
    uint32_t intermediate_slice_3_cb_id = tt::CBIndex::c_11;
    uint32_t intermediate_slice_4_cb_id = tt::CBIndex::c_12;
    uint32_t intermediate_slice_5_cb_id = tt::CBIndex::c_13;
    uint32_t intermediate_slice_6_cb_id = tt::CBIndex::c_14;
    uint32_t intermediate_slice_7_cb_id = tt::CBIndex::c_15;

    uint32_t compute_cb_id = tt::CBIndex::c_16;

    tt::tt_metal::ProgramDescriptor desc;

    // Helper: declare a dynamically-addressed CB bound to a specific buffer.  The
    // framework patches the CB base address on every dispatch from `buffer`'s
    // current allocation (replaces the legacy set_globally_allocated_address +
    // UpdateDynamicCircularBufferAddress dance from override_runtime_arguments).
    auto push_tensor_cb = [&](uint32_t cb_id, uint32_t num_pages, tt::tt_metal::Buffer* buffer) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = num_pages * page_size,
            .core_ranges = worker_core_range_set,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_id),
                .data_format = data_format,
                .page_size = page_size,
            }}},
            .buffer = buffer,
        });
    };

    // input CBs (globally-allocated to input tensor buffers)
    push_tensor_cb(input_slice_0_cb_id, compute_input_cb_num_pages, input_tensors.at(0).buffer());
    push_tensor_cb(input_slice_1_cb_id, compute_input_cb_num_pages, input_tensors.at(1).buffer());
    push_tensor_cb(input_slice_2_cb_id, compute_input_cb_num_pages, input_tensors.at(2).buffer());
    push_tensor_cb(input_slice_3_cb_id, compute_input_cb_num_pages, input_tensors.at(3).buffer());
    push_tensor_cb(input_slice_4_cb_id, compute_input_cb_num_pages, input_tensors.at(4).buffer());
    push_tensor_cb(input_slice_5_cb_id, compute_input_cb_num_pages, input_tensors.at(5).buffer());
    push_tensor_cb(input_slice_6_cb_id, compute_input_cb_num_pages, input_tensors.at(6).buffer());
    push_tensor_cb(input_slice_7_cb_id, compute_input_cb_num_pages, input_tensors.at(7).buffer());

    // intermediate CBs (globally-allocated to intermediate slice tensor buffers)
    push_tensor_cb(intermediate_slice_0_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(0).buffer());
    push_tensor_cb(intermediate_slice_1_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(1).buffer());
    push_tensor_cb(intermediate_slice_2_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(2).buffer());
    push_tensor_cb(intermediate_slice_3_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(3).buffer());
    push_tensor_cb(intermediate_slice_4_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(4).buffer());
    push_tensor_cb(intermediate_slice_5_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(5).buffer());
    push_tensor_cb(intermediate_slice_6_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(6).buffer());
    push_tensor_cb(intermediate_slice_7_cb_id, compute_input_cb_num_pages, intermediate_slice_tensors.at(7).buffer());

    // compute CB (locally allocated; no buffer binding)
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = compute_ouput_cb_num_pages * page_size,
        .core_ranges = worker_core_range_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(compute_cb_id),
            .data_format = data_format,
            .page_size = page_size,
        }}},
    });

    // reader
    std::vector<uint32_t> reader_ct_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        tile_granularity,  // tile_granularity
        input_slice_0_cb_id,
        input_slice_1_cb_id,
        input_slice_2_cb_id,
        input_slice_3_cb_id,
        input_slice_4_cb_id,
        input_slice_5_cb_id,
        input_slice_6_cb_id,
        input_slice_7_cb_id,
        intermediate_slice_0_cb_id,
        intermediate_slice_1_cb_id,
        intermediate_slice_2_cb_id,
        intermediate_slice_3_cb_id,
        intermediate_slice_4_cb_id,
        intermediate_slice_5_cb_id,
        intermediate_slice_6_cb_id,
        intermediate_slice_7_cb_id,
    };

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = worker_core_range_set;
    reader_kernel_desc.compile_time_args = std::move(reader_ct_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    // writer
    std::vector<uint32_t> writer_ct_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        page_size,         // page_size
        tile_granularity,  // tile_granularity
        input_slice_0_cb_id,
        input_slice_1_cb_id,
        input_slice_2_cb_id,
        input_slice_3_cb_id,
        input_slice_4_cb_id,
        input_slice_5_cb_id,
        input_slice_6_cb_id,
        input_slice_7_cb_id,
        compute_cb_id,
    };
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(0).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(1).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(2).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(3).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(4).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(5).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(6).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(7).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = worker_core_range_set;
    writer_kernel_desc.compile_time_args = std::move(writer_ct_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    // reduce
    tt::tt_metal::KernelDescriptor reduce_kernel_desc;
    reduce_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_reduction.cpp";
    reduce_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reduce_kernel_desc.core_ranges = worker_core_range_set;
    reduce_kernel_desc.compile_time_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        tile_granularity,  // tile_granularity
        input_slice_0_cb_id,
        input_slice_1_cb_id,
        input_slice_2_cb_id,
        input_slice_3_cb_id,
        input_slice_4_cb_id,
        input_slice_5_cb_id,
        input_slice_6_cb_id,
        input_slice_7_cb_id,
        intermediate_slice_0_cb_id,
        intermediate_slice_1_cb_id,
        intermediate_slice_2_cb_id,
        intermediate_slice_3_cb_id,
        intermediate_slice_4_cb_id,
        intermediate_slice_5_cb_id,
        intermediate_slice_6_cb_id,
        intermediate_slice_7_cb_id,
        compute_cb_id,
    };
    reduce_kernel_desc.config = tt::tt_metal::ComputeConfigDescriptor{};

    // Push kernel descriptors NOW (before per-link runtime-args loop) so we can
    // refer to them by stable index for both emplace_runtime_args() and the
    // fabric helper, which expects a KernelHandle that indexes into desc.kernels
    // for the ProgramDescriptor overload.
    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(reduce_kernel_desc));
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 1;
    tt::tt_metal::KernelHandle reduce_kernel_id = 2;

    // runtime args
    for (uint32_t link = 0; link < clamped_num_links; link++) {
        for (uint32_t direction = 0; direction < num_directions_per_link; direction++) {
            uint32_t worker_id = (link * num_directions_per_link) + direction;
            uint32_t opposite_direction_worker_id =
                (link * num_directions_per_link) + ((direction + 1) % num_directions_per_link);

            CoreCoord core = worker_cores[worker_id];
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

            CoreCoord opposite_direction_core = worker_cores[opposite_direction_worker_id];
            CoreCoord opposition_direction_virtual_core =
                mesh_device->worker_core_from_logical_core(opposite_direction_core);

            /*
             * NOTE
             * - need to create kernels even if worker not processing tiles, required for pre and post op barrier/sync
             * - min so that we don't try to process non-existent tiles on that dummy worker
             */
            uint32_t start_tiles_read = num_pages_per_shard * worker_id;
            uint32_t start_tiles_to_read = num_pages_per_shard * (worker_id + 1);
            start_tiles_to_read = std::min(start_tiles_to_read, num_pages_per_slice);

            // reader: op_semaphore absolute address is copied on every dispatch
            // via apply_descriptor_runtime_args (the GlobalSemaphore device-side
            // allocation is owned by workload_resources and lives for the
            // lifetime of the cached workload).  GlobalSemaphore::address()
            // returns DeviceAddr (uint64_t); narrow to uint32_t to match the
            // descriptor's variant<uint32_t, Buffer*> arg type — the legacy code
            // narrowed the same way via the std::vector<uint32_t> rt-args buffer.
            tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args_builder;
            reader_rt_args_builder.reserve(4);
            reader_rt_args_builder.push_back(static_cast<uint32_t>(op_semaphore.address()));  // op_semaphore
            reader_rt_args_builder.push_back(direction);                                      // direction
            reader_rt_args_builder.push_back(start_tiles_read);                               // start_tiles_read
            reader_rt_args_builder.push_back(start_tiles_to_read);                            // start_tiles_to_read
            desc.kernels[reader_kernel_id].emplace_runtime_args(core, reader_rt_args_builder);

            // writer: build the rt-arg vector first (so the fabric helper can
            // append its own connection args), then promote to an RTArgList that
            // upgrades the intermediate/output buffer base addresses to Buffer*
            // entries.  The framework records BufferBindings at those positions
            // and patches them on cache hit, removing the need for
            // override_runtime_arguments.
            std::vector<uint32_t> writer_rt_args = {
                intermediate_slice_tensors.at(0).buffer()->address(),  // intermediate_slice_0_address (placeholder)
                intermediate_slice_tensors.at(1).buffer()->address(),  // intermediate_slice_1_address (placeholder)
                intermediate_slice_tensors.at(2).buffer()->address(),  // intermediate_slice_2_address (placeholder)
                intermediate_slice_tensors.at(3).buffer()->address(),  // intermediate_slice_3_address (placeholder)
                intermediate_slice_tensors.at(4).buffer()->address(),  // intermediate_slice_4_address (placeholder)
                intermediate_slice_tensors.at(5).buffer()->address(),  // intermediate_slice_5_address (placeholder)
                intermediate_slice_tensors.at(6).buffer()->address(),  // intermediate_slice_6_address (placeholder)
                intermediate_slice_tensors.at(7).buffer()->address(),  // intermediate_slice_7_address (placeholder)
                output_tensor.buffer()->address(),                     // output_address (placeholder)
                virtual_core.x,                                        // op_semaphore_noc0_x
                virtual_core.y,                                        // op_semaphore_noc0_y
                op_semaphore.address(),                                // op_semaphore
                opposition_direction_virtual_core.x,                   // pre_op_barrier_semaphore_noc0_x
                opposition_direction_virtual_core.y,                   // pre_op_barrier_semaphore_noc0_y
                pre_op_barrier_semaphore.address(),                    // pre_op_barrier_semaphore
                direction,                                             // direction
                start_tiles_read,                                      // start_tiles_read
                start_tiles_to_read,                                   // tiles_to_read
            };

            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            dst_nodes.reserve(1);
            if (direction == 0) {
                // backward
                const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coordinate.value());
                dst_nodes.push_back(backward_coord_fabric_node_id);
            } else {
                // forward
                const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coordinate.value());
                dst_nodes.push_back(forward_coord_fabric_node_id);
            }
            // The fabric helper's ProgramDescriptor specialization indexes into
            // desc.kernels via the KernelHandle to add per-kernel defines and
            // appends additional fabric connection args onto writer_rt_args.
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args<tt::tt_metal::ProgramDescriptor>(
                sender_fabric_node_id, dst_nodes, {link}, desc, writer_kernel_id, core, writer_rt_args);

            // Promote writer RT args to the descriptor.  Indices 0..7 are the
            // intermediate slice buffer addresses; index 8 is the output buffer
            // address — push all nine as Buffer* so the framework records a
            // BufferBinding at each position and patches them on cache hit.  All
            // other positions remain plain uint32_t.
            tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args_builder;
            writer_rt_args_builder.reserve(writer_rt_args.size());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(0).buffer());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(1).buffer());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(2).buffer());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(3).buffer());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(4).buffer());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(5).buffer());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(6).buffer());
            writer_rt_args_builder.push_back(intermediate_slice_tensors.at(7).buffer());
            writer_rt_args_builder.push_back(output_tensor.buffer());
            for (size_t i = 9; i < writer_rt_args.size(); ++i) {
                writer_rt_args_builder.push_back(writer_rt_args[i]);
            }
            desc.kernels[writer_kernel_id].emplace_runtime_args(core, writer_rt_args_builder);

            // reduce
            tt::tt_metal::KernelDescriptor::RTArgList reduce_rt_args_builder;
            reduce_rt_args_builder.reserve(3);
            reduce_rt_args_builder.push_back(start_tiles_read);     // start_tiles_read
            reduce_rt_args_builder.push_back(start_tiles_to_read);  // start_tiles_to_read
            reduce_rt_args_builder.push_back(direction);            // direction
            desc.kernels[reduce_kernel_id].emplace_runtime_args(core, reduce_rt_args_builder);
        }
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
