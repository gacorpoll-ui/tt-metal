// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ProgramDescriptor SendAsyncMeshWorkloadFactory::create_descriptor(
    const SendAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "SendAsyncMeshWorkloadFactory::create_descriptor requires a mesh dispatch coordinate");
    const ttnn::MeshCoordinate mesh_coordinate = mesh_dispatch_coordinate.value();

    auto mesh_socket = operation_attributes.mesh_socket;
    const auto& input_tensor = tensor_args;
    auto* mesh_device = input_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.device();

    ProgramDescriptor desc;
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    std::vector<CoreCoord> sender_core_coords;
    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<size_t> connection_indices;

    for (size_t conn_idx = 0; conn_idx < socket_connection_config.size(); ++conn_idx) {
        const auto& connection = socket_connection_config[conn_idx];
        if (socket_mesh_device->get_device(connection.sender_core.device_coord)->id() == target_device->id()) {
            sender_core_coords.push_back(connection.sender_core.core_coord);
            receiver_core_coords.push_back(connection.receiver_core.core_coord);
            sender_fabric_node_ids.push_back(
                input_tensor.device()->get_fabric_node_id(connection.sender_core.device_coord));
            receiver_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::RECEIVER, connection.receiver_core.device_coord));
            connection_indices.push_back(conn_idx);
        }
    }
    uint32_t num_cores = sender_core_coords.size();

    // Coords without any sender connections to this device get a no-op
    // program.  The legacy create_mesh_workload filtered these out via
    // get_workload_coords; in the descriptor pattern the framework iterates
    // every tensor coord, so return an empty ProgramDescriptor here to
    // preserve the legacy "do nothing on this device" behavior.
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

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_tensor.buffer()->alignment());
    auto input_page_size = input_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);
    auto total_num_pages = input_tensor.buffer()->num_pages();

    uint32_t pages_per_core = total_num_pages / num_cores;
    uint32_t remainder_pages = total_num_pages % num_cores;

    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;

    uint32_t num_whole_packets_per_page = 0, partial_packet_size = 0, socket_block_size = 0;
    if (num_pages_per_packet > 0) {
        socket_block_size = num_pages_per_packet * socket_aligned_page_size;
    } else {
        num_whole_packets_per_page = input_page_size / fabric_max_payload_size;
        partial_packet_size = input_page_size % fabric_max_payload_size;
        socket_block_size = socket_aligned_page_size;
    }

    uint32_t cb_num_pages = 2;
    uint32_t cb_page_size = fabric_max_payload_size;

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    auto src0_cb_index = tt::CBIndex::c_0;

    std::set<CoreRange> sender_core_ranges;
    for (const auto& core : sender_core_coords) {
        sender_core_ranges.insert(CoreRange(core));
    }
    CoreRangeSet sender_core_range_set(sender_core_ranges);

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * cb_page_size,
        .core_ranges = sender_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = cb_page_size,
        }}},
    });

    uint32_t packet_header_cb_num_pages = 2;  // One for data, one for sync
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_1;

    desc.cbs.push_back(CBDescriptor{
        .total_size = packet_header_cb_num_pages * packet_header_cb_page_size,
        .core_ranges = sender_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_header_cb_index),
            .data_format = tt::DataFormat::UInt32,
            .page_size = packet_header_cb_page_size,
        }}},
    });

    bool socket_storage_in_dram =
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer());
    auto compile_time_args = input_accessor_args.get_compile_time_args();
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,               // cb0_id
        input_page_size,             // input_page_size
        socket_aligned_page_size,    // socket_page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // fabric_max_payload_size
    };
    reader_compile_args.insert(reader_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_reader.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = sender_core_range_set;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(reader_kernel_desc));
    KernelHandle reader_kernel_id = desc.kernels.size() - 1;

    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,               // cb0_id
        packet_header_cb_index,      // fabric_packet_header_cb_id
        socket_block_size,           // socket_block_size
        socket_aligned_page_size,    // socket_page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // whole_packet_size (fabric_max_payload_size)
        socket_storage_in_dram,      // is_dram
    };

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_writer.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = sender_core_range_set;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(writer_kernel_desc));
    KernelHandle writer_kernel_id = desc.kernels.size() - 1;

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& sender_core_coord = sender_core_coords[core_idx];
        const auto& receiver_core_coord = receiver_core_coords[core_idx];
        uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

        uint32_t page_start_offset = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);
        uint32_t num_whole_packets = 0, num_pages_remainder = 0;
        if (num_pages_per_packet > 0) {
            num_whole_packets = pages_for_this_core / num_pages_per_packet;
            num_pages_remainder = pages_for_this_core % num_pages_per_packet;
        }

        // Reader RT args.  Pushed as plain uint32_t (no Buffer* bindings)
        // because the writer needs to refresh mesh_socket.get_config_buffer()
        // every dispatch and that buffer is not part of tensor_args.  Mixing
        // bindings with non-binding dynamic addresses is unsafe (the fast
        // path would skip the non-binding positions), so we keep the whole
        // factory on the slow path: framework re-calls create_descriptor on
        // cache hit, which re-reads buffer->address() correctly.
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // input_base_addr
            pages_for_this_core,               // num_pages
            page_start_offset,                 // page_start_offset
            num_whole_packets,                 // num_whole_packets
            num_pages_remainder,               // num_pages_remainder
        };
        desc.kernels[reader_kernel_id].runtime_args.emplace_back(sender_core_coord, std::move(reader_rt_args));

        // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
        uint32_t bank_id = 0;
        if (!socket_storage_in_dram) {
            const auto& connection = socket_connection_config[connection_indices[core_idx]];
            auto* receiver_device = socket_mesh_device->get_device(connection.receiver_core.device_coord);
            bank_id = receiver_device->allocator()->get_bank_ids_from_logical_core(
                mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];
        } else {
            // Assign DRAM banks in round-robin for each receiver core
            auto num_dram_banks = target_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
            bank_id = core_idx % num_dram_banks;
        }

        // Writer RT args.  Pushed as plain uint32_t (no Buffer* bindings):
        // index 0 is the socket config buffer base address, which is not a
        // tensor and therefore can't be bound.  See the reader comment above
        // for the rationale.  Slow path correctly refreshes both addresses
        // on every dispatch.
        std::vector<uint32_t> writer_rt_args = {
            mesh_socket.get_config_buffer()->address(),  // socket_config_addr
            bank_id,                                     // bank_id
            pages_for_this_core,                         // num_pages
            page_start_offset,                           // page_start_offset
            num_whole_packets,                           // num_whole_packets
            num_pages_remainder,                         // num_pages_remainder
        };

        const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];
        auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);

        uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            selected_link_index,
            desc,
            sender_core_coord,
            writer_rt_args);

        desc.kernels[writer_kernel_id].runtime_args.emplace_back(sender_core_coord, std::move(writer_rt_args));
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
