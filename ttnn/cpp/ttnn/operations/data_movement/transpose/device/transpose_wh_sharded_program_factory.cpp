// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TransposeWHShardedProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    IDevice* device = input_tensor.device();

    const bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    const auto shard_spec = input_tensor.shard_spec().value();
    const bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    const auto& all_cores = shard_spec.grid;
    const uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();

    ProgramDescriptor desc;

    // --- CB descriptors (dynamic buffers: framework refreshes address for sharded cached programs) ---
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_shard * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
        .buffer = src_buffer,
    });

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_shard * dst_single_tile_size,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
        .buffer = dst_buffer,
    });

    // --- Compute parameters ---
    const auto padded_shape = input_tensor.padded_shape();
    const auto shard_shape = shard_spec.shape;

    const uint32_t H = padded_shape[2];
    const uint32_t Hs = shard_shape[0];
    const uint32_t Ws = shard_shape[1];

    const uint32_t Hts = Hs / tile.get_height();
    const uint32_t Wts = Ws / tile.get_width();

    const uint32_t Ht = H / tile.get_height();
    const uint32_t Ht_per_shard = std::min(Ht, Hts);

    const uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    const uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    const uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    // --- Per-core runtime args ---
    const auto bbox = all_cores.bounding_box();
    const std::vector<CoreCoord> cores =
        grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_cores;
    reader_desc.compile_time_args = {src0_cb_index};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = total_cores;
    writer_desc.compile_time_args = {output_cb_index};
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_sharded.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = total_cores;
    compute_desc.compile_time_args = {src0_cb_index, output_cb_index};
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
    };

    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());

    const uint32_t num_active_cores = all_cores.num_cores();
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        if (i < num_active_cores) {
            reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_blocks});
            writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_blocks});
            compute_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    num_blocks, HtWt_tile_size, num_hw_blocks_per_shard, Ht_per_shard, Wts});
        } else {
            // Noop cores still need runtime args set (sized to original signatures).
            reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(1, 0));
            writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(1, 0));
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(5, 0));
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
