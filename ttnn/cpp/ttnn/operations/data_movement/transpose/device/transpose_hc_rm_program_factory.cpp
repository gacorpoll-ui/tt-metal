// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_rm_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TransposeHCRMProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto& a_shape = input_tensor.logical_shape();
    const uint32_t W = a_shape[3];
    const uint32_t H = a_shape[2];
    const uint32_t C = a_shape[1];
    const uint32_t N = a_shape[0];
    const uint32_t NCH = N * C * H;

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    log_debug(tt::LogOp, "transpose_hc_rm");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);

    IDevice* device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, NCH);

    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    constexpr uint32_t src0_cb_index = 0;

    const uint32_t num_sticks = std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    const uint32_t aligned_page = std::max(src0_buffer->aligned_page_size(), dst_buffer->aligned_page_size());
    const uint32_t stick_size = std::max(W * input_tensor.element_size(), aligned_page);

    ProgramDescriptor desc;

    // --- CB descriptor ---
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_sticks * stick_size,
        .core_ranges = CoreRangeSet(total_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = stick_size,
        }}},
    });

    // --- Reader kernel descriptor ---
    std::vector<uint32_t> reader_compile_time_args = {N, H, C, stick_size, src0_buffer->aligned_page_size()};
    std::vector<uint32_t> reader_common_runtime_args;
    TensorAccessorArgs(*src0_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet(total_cores);
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.common_runtime_args = std::move(reader_common_runtime_args);

    // --- Writer kernel descriptor ---
    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index, stick_size, dst_buffer->aligned_page_size()};
    std::vector<uint32_t> writer_common_runtime_args;
    TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_interleaved_start_id_rm.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = CoreRangeSet(total_cores);
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.common_runtime_args = std::move(writer_common_runtime_args);

    // --- Per-core runtime args ---
    const uint32_t W_bytes = W * input_tensor.element_size();
    constexpr uint32_t max_read_size = 2048;
    uint32_t curr_c = 0;
    uint32_t curr_h = 0;
    uint32_t curr_n = 0;

    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; ++i) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        }

        uint32_t num_sticks_per_core_read = 0;
        uint32_t num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        // Buffer* entries auto-register as BufferBindings → framework patches addresses on cache hits.
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer, num_sticks_per_core_read, num_read_per_barrier, curr_sticks_read, curr_c, curr_h, curr_n});
        writer_desc.emplace_runtime_args(
            core, {dst_buffer, num_sticks_per_core_read, num_read_per_barrier, curr_sticks_write});

        curr_sticks_write += num_sticks_per_core;
        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            ++curr_c;
            curr_sticks_read += H;
            if (curr_c == C) {
                ++curr_h;
                curr_c = 0;
                if (curr_h == H) {
                    ++curr_n;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
