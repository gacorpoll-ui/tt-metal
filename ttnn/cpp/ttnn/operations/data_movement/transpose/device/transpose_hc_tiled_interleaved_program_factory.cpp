// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/math.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace ttnn::prim {

ProgramDescriptor TransposeHCTiledInterleavedProgramFactory::create_descriptor(
    const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    const float pad_value = operation_attributes.pad_value;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto tile = input_tensor.tensor_spec().tile();
    const auto tile_shape = tile.get_tile_shape();
    const auto face_shape = tile.get_face_shape();
    const uint32_t C = input_tensor.logical_shape()[1];
    const uint32_t H = input_tensor.logical_shape()[2];
    const uint32_t W = input_tensor.logical_shape()[3];
    const bool needs_padding = (C % tile_shape[1] != 0);

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    const uint32_t element_size = input_tensor.element_size();

    const uint32_t max_padding_write = face_shape[0] * face_shape[1];

    ProgramDescriptor desc;

    // --- CB descriptors ---
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t padding_cb_index = tt::CBIndex::c_1;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * single_tile_size,
        .core_ranges = CoreRangeSet(total_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    if (needs_padding) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = max_padding_write * element_size,
            .core_ranges = CoreRangeSet(total_cores),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(padding_cb_index),
                .data_format = cb_data_format,
                .page_size = max_padding_write * element_size,
            }}},
        });
    }

    // --- Padding value packing ---
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    if (needs_padding) {
        const uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = max_padding_write / num_packed_values;
        switch (input_tensor.dtype()) {
            case DataType::INT32:
            case DataType::UINT32: padding_val_packed = pad_value; break;
            case DataType::BFLOAT16:
                padding_val_packed = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
                break;
            case DataType::UINT16:
                padding_val_packed =
                    pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
                break;
            case DataType::FLOAT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            default:
                padding_val_packed = 0;
                TT_FATAL(
                    false,
                    "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                    "FLOAT32");
        }
    }

    // --- Reader kernel descriptor ---
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> reader_common_runtime_args;
    TensorAccessorArgs(*src_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet(total_cores);
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.named_compile_time_args = {
        {"num_writes", num_writes},
        {"padding_val_packed", padding_val_packed},
        {"needs_padding", static_cast<uint32_t>(needs_padding)},
        {"swap_hw", 0u},
        {"H", 1u},
        {"W", 1u},
        {"accumulated_outer_dims", 1u},
        {"tile_height", 1u},
        {"tile_width", 1u},
    };
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.common_runtime_args = std::move(reader_common_runtime_args);

    // --- Writer kernel descriptor ---
    std::vector<uint32_t> writer_compile_time_args = {
        element_size,
        tt::CBIndex::c_0,
        C,
        H,
        W,
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1],
        static_cast<uint32_t>(needs_padding)};
    std::vector<uint32_t> writer_common_runtime_args;
    TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = CoreRangeSet(total_cores);
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.common_runtime_args = std::move(writer_common_runtime_args);

    // --- Per-core runtime args ---
    const auto tile_hw = tile_shape[0] * tile_shape[1];
    const uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    const uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    // Only last row of Ct should have padding; reader walks the input tile grid while writer walks
    // the (potentially padded) output grid, so we pre-compute two splits.
    const uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[2] / tile_shape[0]);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    uint32_t start_idx = 0;
    uint32_t padded_start_idx = 0;
    for (const auto& core : total_cores) {
        uint32_t num_tiles_per_core = 0;
        uint32_t padded_tiles_per_core = 0;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        }

        if (padded_core_group_1.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_1;
        } else if (padded_core_group_2.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_2;
        }

        const uint32_t end_idx = start_idx + num_tiles_per_core;
        const uint32_t padded_end_idx = padded_start_idx + padded_tiles_per_core;

        // Buffer* entries auto-register as BufferBindings → framework patches addresses on cache hits.
        reader_desc.emplace_runtime_args(core, {src_buffer, num_tiles_per_core, start_idx});
        writer_desc.emplace_runtime_args(core, {dst_buffer, start_idx, end_idx, padded_start_idx, padded_end_idx});

        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
