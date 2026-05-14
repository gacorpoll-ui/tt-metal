// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor RecvAsyncMeshWorkloadFactory::create_descriptor(
    const RecvAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "RecvAsyncMeshWorkloadFactory::create_descriptor requires a mesh dispatch coordinate");
    const ttnn::MeshCoordinate mesh_coordinate = mesh_dispatch_coordinate.value();

    auto mesh_socket = operation_attributes.mesh_socket;
    const auto& output_tensor = tensor_args;
    auto* mesh_device = output_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.device();

    ProgramDescriptor desc;

    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<uint32_t> connection_indices;

    // TODO #24995: Find appropriate receiver cores and fabric node IDs based on mesh socket configuration
    for (uint32_t i = 0; i < socket_connection_config.size(); ++i) {
        const auto& connection = socket_connection_config[i];
        if (socket_mesh_device->get_device(connection.receiver_core.device_coord)->id() == target_device->id()) {
            receiver_core_coords.push_back(connection.receiver_core.core_coord);
            receiver_fabric_node_ids.push_back(
                output_tensor.device()->get_fabric_node_id(connection.receiver_core.device_coord));
            sender_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::SENDER, connection.sender_core.device_coord));
            connection_indices.push_back(i);
        }
    }
    uint32_t num_cores = receiver_core_coords.size();

    // No-op program for non-receiver coords (mirrors the legacy get_workload_coords<RECEIVER>
    // filtering and the send_async migration pattern).
    if (num_cores == 0) {
        return desc;
    }

    // cores must not exceed available fabric links
    {
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[0];
        const auto& sender_fabric_node_id = sender_fabric_node_ids[0];
        auto available_link_indices =
            tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
        uint32_t num_available_links = available_link_indices.size();

        TT_FATAL(
            num_cores <= num_available_links,
            "Cannot create {} receiver-sender pairs with only {} available fabric links between devices. "
            "Reduce the number of cores per device. "
            "Available links: {}, Requested pairs: {}",
            num_cores,
            num_available_links,
            num_available_links,
            num_cores);
    }

    // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        output_tensor.buffer()->alignment());
    auto output_page_size = output_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(output_page_size, max_alignment);
    auto total_num_pages = output_tensor.buffer()->num_pages();
    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;

    uint32_t pages_per_core = total_num_pages / num_cores;
    uint32_t remainder_pages = total_num_pages % num_cores;

    uint32_t socket_block_size = 0;
    if (num_pages_per_packet > 0) {
        socket_block_size = num_pages_per_packet * socket_aligned_page_size;
    } else {
        socket_block_size = socket_aligned_page_size;
    }

    auto receiver_core_range_set = CoreRangeSet(std::set<CoreRange>());
    for (const auto& core : receiver_core_coords) {
        receiver_core_range_set = receiver_core_range_set.merge(CoreRangeSet({CoreRange(core, core)}));
    }

    uint32_t packet_header_cb_num_pages = 1;  // One for sync
    uint32_t packet_header_cb_page_size = fabric_max_payload_size;

    auto packet_header_cb_index = tt::CBIndex::c_0;

    desc.cbs.push_back(CBDescriptor{
        .total_size = packet_header_cb_num_pages * packet_header_cb_page_size,
        .core_ranges = receiver_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_header_cb_index),
            .data_format = tt::DataFormat::UInt32,
            .page_size = packet_header_cb_page_size,
        }}},
    });

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer());
    auto output_accessor_compile_time_args = output_accessor_args.get_compile_time_args();

    tt::CBIndex scratch_buffer_cb_index = tt::CBIndex::c_1;
    bool socket_storage_in_dram =
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;

    if (socket_storage_in_dram) {
        // For DRAM mode, scratch buffer size should be based on packet size, not total pages per core
        // This matches the original single-core logic: 2 * num_pages_per_block * socket_aligned_page_size
        uint32_t num_pages_per_block = 0;
        if (num_pages_per_packet > 0) {
            num_pages_per_block = num_pages_per_packet;
        } else {
            num_pages_per_block = 1;
        }
        uint32_t scratch_buffer_size = 2 * num_pages_per_block * socket_aligned_page_size;

        auto data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
        desc.cbs.push_back(CBDescriptor{
            .total_size = scratch_buffer_size,
            .core_ranges = receiver_core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(scratch_buffer_cb_index),
                .data_format = data_format,
                .page_size = socket_aligned_page_size,
            }}},
        });
    }

    if (!socket_storage_in_dram) {
        std::vector<uint32_t> writer_compile_args = {
            packet_header_cb_index,    // fabric_packet_header_cb_id
            output_page_size,          // output_page_size
            socket_block_size,         // socket_block_size
            socket_aligned_page_size,  // socket_page_size
            num_pages_per_packet,      // num_pages_per_packet
        };
        writer_compile_args.insert(
            writer_compile_args.end(),
            output_accessor_compile_time_args.begin(),
            output_accessor_compile_time_args.end());

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/"
            "receiver_inplace_writer.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = receiver_core_range_set;
        writer_desc.compile_time_args = std::move(writer_compile_args);
        writer_desc.config = WriterConfigDescriptor{};

        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            const auto& receiver_core_coord = receiver_core_coords[core_idx];
            const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
            const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];

            uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

            uint32_t page_start_offset = 0;
            for (uint32_t prev_idx = 0; prev_idx < core_idx; ++prev_idx) {
                uint32_t prev_pages = pages_per_core + (prev_idx < remainder_pages ? 1 : 0);
                page_start_offset += prev_pages;
            }

            uint32_t num_whole_packets = 0, num_pages_remainder = 0;
            if (num_pages_per_packet > 0) {
                num_whole_packets = pages_for_this_core / num_pages_per_packet;
                num_pages_remainder = pages_for_this_core % num_pages_per_packet;
            }

            std::vector<uint32_t> writer_rt_args = {
                mesh_socket.get_config_buffer()->address(),  // socket_config_addr
                output_tensor.buffer()->address(),           // output_base_addr
                pages_for_this_core,                         // num_pages
                page_start_offset,                           // page_start_offset
                num_whole_packets,                           // num_whole_packets
                num_pages_remainder,                         // num_pages_remainder
            };

            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                desc,
                receiver_core_coord,
                writer_rt_args);

            writer_desc.runtime_args.emplace_back(receiver_core_coord, std::move(writer_rt_args));
        }

        desc.kernels.push_back(std::move(writer_desc));
    } else {
        std::vector<uint32_t> reader_compile_args = {
            packet_header_cb_index,                         // fabric_packet_header_cb_id
            scratch_buffer_cb_index,                        // scratch_buffer_cb_id
            socket_block_size,                              // socket_block_size
            socket_aligned_page_size,                       // socket_page_size
            static_cast<uint32_t>(socket_storage_in_dram),  // socket_storage_in_dram
        };

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_reader.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = receiver_core_range_set;
        reader_desc.compile_time_args = std::move(reader_compile_args);
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_compile_args = {
            scratch_buffer_cb_index,  // scratch_buffer_cb_id
            output_page_size,         // page_size
        };
        writer_compile_args.insert(
            writer_compile_args.end(),
            output_accessor_compile_time_args.begin(),
            output_accessor_compile_time_args.end());

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_writer.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = receiver_core_range_set;
        writer_desc.compile_time_args = std::move(writer_compile_args);
        writer_desc.config = WriterConfigDescriptor{};

        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            const auto& receiver_core_coord = receiver_core_coords[core_idx];
            const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
            const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];

            uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

            uint32_t page_start_offset = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);

            uint32_t num_whole_packets = 0, num_pages_remainder_core = 0;
            if (num_pages_per_packet > 0) {
                num_whole_packets = pages_for_this_core / num_pages_per_packet;
                num_pages_remainder_core = pages_for_this_core % num_pages_per_packet;
            }

            uint32_t num_blocks = 0, num_pages_per_block = 0, block_remainder_pages = 0;
            if (num_pages_per_packet > 0) {
                num_blocks = num_whole_packets;
                num_pages_per_block = num_pages_per_packet;
                block_remainder_pages = num_pages_remainder_core;
            } else {
                num_blocks = pages_for_this_core;
                num_pages_per_block = 1;
                block_remainder_pages = 0;
            }

            // TODO #24995: This should be derived from the expected tensor/socket configuration
            uint32_t bank_id = 0;
            if (socket_storage_in_dram) {
                // Assign DRAM banks in round-robin for each receiver core
                auto num_dram_banks = target_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
                bank_id = core_idx % num_dram_banks;
            } else {
                // L1 mode: use logical core mapping
                bank_id = target_device->allocator()->get_bank_ids_from_logical_core(
                    mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];
            }

            std::vector<uint32_t> reader_rt_args = {
                mesh_socket.get_config_buffer()->address(),  // socket_config_addr
                bank_id,                                     // bank_id
                num_blocks,                                  // num_blocks
                num_pages_per_block,                         // num_pages_per_block
                block_remainder_pages,                       // block_remainder_pages
            };

            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];

            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                desc,
                receiver_core_coord,
                reader_rt_args);

            reader_desc.runtime_args.emplace_back(receiver_core_coord, std::move(reader_rt_args));

            std::vector<uint32_t> writer_rt_args = {
                output_tensor.buffer()->address(),  // output_base_addr
                page_start_offset,                  // start_page_index
                pages_for_this_core,                // num_pages
            };

            writer_desc.runtime_args.emplace_back(receiver_core_coord, std::move(writer_rt_args));
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
