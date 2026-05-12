// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TransposeHCTiledProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const uint32_t sub_tile_line_bytes = 16 * input_tensor.element_size();
    const uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "transpose_hc_tiled");
    log_debug(tt::LogOp, "sub_tile_line_bytes: {}", sub_tile_line_bytes);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);

    IDevice* device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Kernel reads 16-element face lines (32 B for BFLOAT16) and may need to copy via a scratch CB when the buffer
    // alignment (64 B on Blackhole) exceeds the face-line size.
    const uint32_t alignment = dst_buffer->alignment();
    const bool misaligned = alignment > sub_tile_line_bytes;

    ProgramDescriptor desc;

    // --- CB descriptors ---
    constexpr uint32_t src0_cb_index = 0;
    constexpr uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = CoreRangeSet(total_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    if (misaligned) {
        constexpr uint32_t src1_cb_index = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = alignment,
            .core_ranges = CoreRangeSet(total_cores),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src1_cb_index),
                .data_format = cb_data_format,
                .page_size = alignment,
            }}},
        });
    }

    // --- Reader kernel descriptor ---
    std::vector<uint32_t> reader_compile_time_args = {
        sub_tile_line_bytes, static_cast<uint32_t>(cb_data_format == tt::DataFormat::Float32 ? 1 : 0), alignment};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_partitioned.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet(total_cores);
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // --- Writer kernel descriptor ---
    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = CoreRangeSet(total_cores);
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // --- Per-core runtime args ---
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t W = input_shape[3];
    const uint32_t H = input_shape[2];
    const uint32_t C = input_shape[1];
    const uint32_t HW_bytes = H * W * input_tensor.element_size();
    const uint32_t CHW_bytes = C * H * W * input_tensor.element_size();
    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ct = C / TILE_HEIGHT;
    const uint32_t CtHWt = Ct * H * Wt;
    const uint32_t CtWt = Ct * Wt;

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        }

        const uint32_t h = num_tiles_read / CtWt % H;
        const uint32_t ct = num_tiles_read / Wt % Ct;

        // Buffer* entries auto-register as BufferBindings → framework patches addresses on cache hits.
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,
             Wt,
             H,
             Ct,
             HW_bytes,
             CHW_bytes,
             num_tiles_read,
             num_tiles_per_core,
             num_tiles_read / CtHWt * CHW_bytes,
             h,
             h / TILE_HEIGHT * Wt,
             ct,
             ct * TILE_HEIGHT * HW_bytes,
             num_tiles_read % Wt});
        writer_desc.emplace_runtime_args(core, {dst_buffer, num_tiles_per_core, num_tiles_read});

        num_tiles_read += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
