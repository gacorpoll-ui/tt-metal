// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "tt_metal/test_utils/bfloat_utils.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/float8_utils.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::fp8_typecast {

// Run a datacopy kernel with different input/output CB formats.
// The hardware unpacker reads input_fmt and the packer writes output_fmt,
// performing the format conversion implicitly. fp32_dest_acc_en controls
// whether the Dest register operates in 32-bit mode.
static vector<uint32_t> run_fp8_typecast(
    IDevice* dev,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    const vector<uint32_t>& src_vec,
    uint32_t num_tiles,
    bool fp32_dest_acc_en) {
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t input_tile_size = tt::tile_size(input_fmt);
    uint32_t output_tile_size = tt::tile_size(output_fmt);

    InterleavedBufferConfig src_config{
        .device = dev,
        .size = num_tiles * input_tile_size,
        .page_size = num_tiles * input_tile_size,
        .buffer_type = BufferType::DRAM};
    auto src_buffer = CreateBuffer(src_config);

    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = num_tiles * output_tile_size,
        .page_size = num_tiles * output_tile_size,
        .buffer_type = BufferType::DRAM};
    auto dst_buffer = CreateBuffer(dst_config);

    CircularBufferConfig cb_src_config = CircularBufferConfig(input_tile_size, {{tt::CBIndex::c_0, input_fmt}})
                                             .set_page_size(tt::CBIndex::c_0, input_tile_size);
    CreateCircularBuffer(program, core, cb_src_config);

    CircularBufferConfig cb_dst_config = CircularBufferConfig(output_tile_size, {{tt::CBIndex::c_16, output_fmt}})
                                             .set_page_size(tt::CBIndex::c_16, output_tile_size);
    CreateCircularBuffer(program, core, cb_dst_config);

    auto reader = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_fp8.cpp",
        core,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {num_tiles}});

    detail::WriteToBuffer(src_buffer, src_vec);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles});

    detail::LaunchProgram(dev, program);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// Data generators use the existing uniform-distribution helpers:
//   create_random_vector_of_bfloat16  (bfloat16.hpp)
//   tt::test_utils::create_random_vector_of_bfp8  (bfloat_utils.hpp)
// Both generate U(0, rand_max_float) + offset, so passing rand_max_float=20
// and offset=-10 yields U(-10, 10).
//
// Format-to-float unpackers (fp8_to_floats, bf16_to_floats) live in
// float8_utils.hpp. Validation reuses tt::test_utils::is_close_vectors
// (rtol/atol overload) and check_pcc from comparison.hpp. Only the bfp8
// unpacker is local since it depends on the bfp8-specific tile layout.

static vector<float> bfp8_to_floats(const vector<uint32_t>& packed) {
    return unpack_bfp8_tiles_into_float_vec(
        tt::stl::make_const_span(packed), /*row_major_output=*/false, /*is_exp_a=*/false);
}

// Format-dispatched stimulus generator for the typecast tests. All callers
// share the same uniform-distribution parameters (U(-10, 10), seed=42), so
// these are baked in rather than passed through.
static vector<uint32_t> generate_src(tt::DataFormat fmt, uint32_t num_tiles) {
    const auto bytes = tt::tile_size(fmt) * num_tiles;
    switch (fmt) {
        case tt::DataFormat::Fp8_e4m3:
            return create_random_vector_of_float8_e4m3(bytes, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
        case tt::DataFormat::Float16_b:
            return create_random_vector_of_bfloat16(bytes, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
        case tt::DataFormat::Bfp8_b:
            return tt::test_utils::create_random_vector_of_bfp8(
                bytes, /*is_exp_a=*/false, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
        default: TT_FATAL(false, "generate_src: unsupported format {}", static_cast<int>(fmt));
    }
    return {};
}

// Format-dispatched packed→floats unpacker for the supported test formats.
static vector<float> unpack_to_floats(tt::DataFormat fmt, const vector<uint32_t>& packed) {
    switch (fmt) {
        case tt::DataFormat::Fp8_e4m3: return tt::test_utils::fp8_to_floats(packed);
        case tt::DataFormat::Float16_b: return tt::test_utils::bf16_to_floats(packed);
        case tt::DataFormat::Bfp8_b: return bfp8_to_floats(packed);
        default: TT_FATAL(false, "unpack_to_floats: unsupported format {}", static_cast<int>(fmt));
    }
    return {};
}

// Single body for every typecast test: generate, run, unpack, compare.
// Each TEST_F below is a thin one-line wrapper specifying its own (in_fmt,
// out_fmt, fp32_dest_acc_en, rtol, atol, min_pcc).
static void check_typecast(
    IDevice* dev,
    tt::DataFormat in_fmt,
    tt::DataFormat out_fmt,
    bool fp32_dest_acc_en,
    float rtol,
    float atol,
    double min_pcc) {
    constexpr uint32_t num_tiles = 64;
    auto src_vec = generate_src(in_fmt, num_tiles);
    auto result_vec = run_fp8_typecast(dev, in_fmt, out_fmt, src_vec, num_tiles, fp32_dest_acc_en);
    auto src_floats = unpack_to_floats(in_fmt, src_vec);
    auto dst_floats = unpack_to_floats(out_fmt, result_vec);
    EXPECT_TRUE(tt::test_utils::is_close_vectors<float>(src_floats, dst_floats, rtol, atol));
    EXPECT_TRUE(tt::test_utils::check_pcc(src_floats, dst_floats, min_pcc));
}

}  // namespace unit_tests::llk::fp8_typecast

using namespace unit_tests::llk::fp8_typecast;

// ============================================================================
// fp8_e4m3 → Float16_b
// Widening conversion: every fp8 value is exactly representable in BF16.
// Expected: no precision loss → rtol=0.0, atol=0.0.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToFloat16b) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
        /*fp32_dest_acc_en=*/false,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0);
}

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToFloat16bFp32Dest) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
        /*fp32_dest_acc_en=*/true,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0);
}

// ============================================================================
// Float16_b → fp8_e4m3
// Narrowing: BF16 has 7 mantissa bits vs fp8's 3 → precision loss expected.
// rtol=0.125 covers the max relative quantization error of fp8 (~1/8).
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFloat16bToFp8e4m3) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Float16_b,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/false,
        /*rtol=*/0.125f,
        /*atol=*/0.015625f,
        /*min_pcc=*/0.999);
}

TEST_F(BlackholeSingleCardFixture, TensixFloat16bToFp8e4m3Fp32Dest) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Float16_b,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true,
        /*rtol=*/0.125f,
        /*atol=*/0.015625f,
        /*min_pcc=*/0.999);
}

// ============================================================================
// fp8_e4m3 → Bfp8_b
// Widening: Bfp8_b has 8 mantissa bits and a shared exponent per 16-element
// row. For test data within [-10, 10], fp8 values may lose significant
// precision due to the blocking forming process.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToBfp8b) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Bfp8_b,
        /*fp32_dest_acc_en=*/false,
        /*rtol=*/0.3f,
        /*atol=*/0.3f,
        /*min_pcc=*/0.9999);
}

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToBfp8bFp32Dest) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Bfp8_b,
        /*fp32_dest_acc_en=*/true,
        /*rtol=*/0.3f,
        /*atol=*/0.3f,
        /*min_pcc=*/0.9999);
}

// ============================================================================
// Bfp8_b → fp8_e4m3
// Narrowing: Bfp8_b has 8 mantissa bits vs fp8's 3 → precision loss expected.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixBfp8bToFp8e4m3) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Bfp8_b,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/false,
        /*rtol=*/0.125f,
        /*atol=*/0.015625f,
        /*min_pcc=*/0.999);
}

TEST_F(BlackholeSingleCardFixture, TensixBfp8bToFp8e4m3Fp32Dest) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Bfp8_b,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true,
        /*rtol=*/0.125f,
        /*atol=*/0.015625f,
        /*min_pcc=*/0.999);
}

// ============================================================================
// Bfp8_b → Bfp8_b (identity)
// Same format on both sides. The unpack→Dest→repack round-trip through the
// shared-exponent blocking process may introduce minor rounding, but PCC
// should remain very high.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixBfp8bToBfp8b) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Bfp8_b,
        tt::DataFormat::Bfp8_b,
        /*fp32_dest_acc_en=*/false,
        /*rtol=*/0.3f,
        /*atol=*/0.3f,
        /*min_pcc=*/0.9999);
}

TEST_F(BlackholeSingleCardFixture, TensixBfp8bToBfp8bFp32Dest) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Bfp8_b,
        tt::DataFormat::Bfp8_b,
        /*fp32_dest_acc_en=*/true,
        /*rtol=*/0.3f,
        /*atol=*/0.3f,
        /*min_pcc=*/0.9999);
}

// ============================================================================
// fp8_e4m3 → fp8_e4m3 (identity)
// Same format on both sides. The round-trip should be lossless since every
// fp8 value survives the unpack→Dest→repack cycle exactly.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToFp8e4m3) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/false,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0);
}

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToFp8e4m3Fp32Dest) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0);
}

// ============================================================================
// Float16_b → Float16_b (identity)
// Same format on both sides. The round-trip should be lossless since every
// BF16 value survives the unpack→Dest→repack cycle exactly.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFloat16bToFloat16b) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        /*fp32_dest_acc_en=*/false,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0);
}

TEST_F(BlackholeSingleCardFixture, TensixFloat16bToFloat16bFp32Dest) {
    check_typecast(
        devices_[0]->get_devices()[0],
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        /*fp32_dest_acc_en=*/true,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0);
}

}  // namespace tt::tt_metal
