// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_dot {

tt::tt_metal::ProgramDescriptor MorehDotOperation::SingleCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto* src0_buffer = input_a.buffer();
    auto* src1_buffer = input_b.buffer();
    auto* dst_buffer = output.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_a.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);

    uint32_t num_tiles = input_a.physical_volume() / tt::constants::TILE_HW;
    const auto& a_shape_wo_padding = input_a.logical_shape();
    uint32_t pad_h = a_shape_wo_padding[2] % tt::constants::TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % tt::constants::TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (tt::constants::TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (tt::constants::TILE_WIDTH) : (pad_w);

    tt::tt_metal::IDevice* device = input_a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t in0_t = 2;   // a
    const uint32_t in1_t = 2;   // b
    const uint32_t in2_t = 1;   // scaler
    const uint32_t out0_t = 2;  // out
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    CoreCoord core = {0, 0};
    CoreRangeSet core_range = CoreRangeSet(CoreRange(core, core));

    ProgramDescriptor desc;

    // ---- Circular buffers ----
    auto make_cb = [&](uint8_t cb_idx, uint32_t num_tiles) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * single_tile_size,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_idx,
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    };
    make_cb(static_cast<uint8_t>(CBIndex::c_0), in0_t);
    make_cb(static_cast<uint8_t>(CBIndex::c_1), in1_t);
    make_cb(static_cast<uint8_t>(CBIndex::c_2), in2_t);
    make_cb(static_cast<uint8_t>(CBIndex::c_16), out0_t);
    make_cb(static_cast<uint8_t>(CBIndex::c_24), im0_t);
    make_cb(static_cast<uint8_t>(CBIndex::c_25), im1_t);

    // ---- Reader kernel ----
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {};
    TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/reader_moreh_dot.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.emplace_runtime_args(core, {src0_buffer, src1_buffer, num_tiles, 0u, mask_h, mask_w});

    // ---- Writer kernel ----
    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {static_cast<uint32_t>(CBIndex::c_16)};
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/writer_moreh_dot.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.emplace_runtime_args(core, {output.buffer(), 1u, 0u});

    // ---- Compute kernel ----
    KernelDescriptor::Defines compute_defines = {
        {"REDUCE_OP", "PoolType::SUM"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_ROW"},
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/moreh_dot.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_range;
    compute_desc.compile_time_args = {};
    compute_desc.defines = std::move(compute_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };
    compute_desc.emplace_runtime_args(core, {num_tiles, 1u});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_dot
