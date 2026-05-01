// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/float8.hpp>

namespace tt::test_utils {

namespace detail {
// Element-wise static_cast of any vector of float-convertible values into a
// flat std::vector<float>. Shared body of fp8_to_floats / bf16_to_floats so
// the two public helpers differ only by which unpacker they invoke.
template <typename T>
inline std::vector<float> to_floats(const std::vector<T>& vec) {
    std::vector<float> floats;
    floats.reserve(vec.size());
    for (const auto& v : vec) {
        floats.push_back(static_cast<float>(v));
    }
    return floats;
}
}  // namespace detail

// Unpack a packed uint32 vector (4 fp8 bytes per word) into a flat float vector.
inline std::vector<float> fp8_to_floats(const std::vector<uint32_t>& packed) {
    return detail::to_floats(unpack_uint32_vec_into_float8_e4m3_vec(packed));
}

// Unpack a packed uint32 vector (2 bf16 per word) into a flat float vector.
inline std::vector<float> bf16_to_floats(const std::vector<uint32_t>& packed) {
    return detail::to_floats(unpack_uint32_vec_into_bfloat16_vec(packed));
}

// Linear byte index within a 32x32 tile of 1-byte elements stored as 4 16x16
// faces in (TL, TR, BL, BR) order, addressed by logical (col, row).
// Applies to any 8-bit tile format (Fp8_e4m3, Lf8, Int8, UInt8).
inline int byte_tile_face_major_index(int col, int row) {
    int offset = ((col < 16) ? 0 : 256) + ((row < 16) ? 0 : 512);
    return offset + ((row % 16) * 16) + (col % 16);
}

// Generate `num_tiles` tiles of fp8_e4m3 stimulus drawn from U(-rng, +rng),
// returning both the packed-uint32 device-ready buffer and the matching flat
// float vector (face-major byte order) suitable for golden computation.
inline std::pair<std::vector<uint32_t>, std::vector<float>> make_fp8_input(
    int seed, float rng, std::size_t num_tiles = 1) {
    constexpr std::size_t kTileBytes = 32 * 32;
    const std::size_t total_bytes = kTileBytes * num_tiles;
    auto packed = create_random_vector_of_float8_e4m3(
        total_bytes, /*rand_max_float=*/static_cast<int>(2 * rng), seed, /*offset=*/-rng);
    auto floats = fp8_to_floats(packed);
    return {std::move(packed), std::move(floats)};
}

}  // namespace tt::test_utils
