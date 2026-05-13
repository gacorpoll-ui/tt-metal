// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>
#include <utility>

#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu_types.h"

// Dispatch the SFPU callable with or without leading dst indices, depending on
// which signature it accepts. Some calculate functions take (dst_in, dst_out,
// args...) — these are the split-aware ones added on top of the legacy
// (args...) ones (e.g. TopK helpers). This helper selects the right shape at
// compile time so the single-idx params overload can drive both kinds of
// callable while keeping in == out for the legacy single-dst contract.
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_dispatch_(Callable&& sfpu_func, std::uint32_t dst_index, Args&&... args)
{
    if constexpr (std::is_invocable_v<Callable&, std::uint32_t, std::uint32_t, Args&...>)
    {
        std::forward<Callable>(sfpu_func)(dst_index, dst_index, std::forward<Args>(args)...);
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
}

template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index, int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args)
{
    _llk_math_eltwise_unary_sfpu_start_(dst_index);

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::R)
    {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            _llk_math_eltwise_unary_sfpu_dispatch_(std::forward<Callable>(sfpu_func), dst_index, std::forward<Args>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
        // Skip next two faces
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    }
    else if (mode == VectorMode::C)
    {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            _llk_math_eltwise_unary_sfpu_dispatch_(std::forward<Callable>(sfpu_func), dst_index, std::forward<Args>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else if (mode == VectorMode::RC)
    {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            _llk_math_eltwise_unary_sfpu_dispatch_(std::forward<Callable>(sfpu_func), dst_index, std::forward<Args>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_dispatch_(std::forward<Callable>(sfpu_func), dst_index, std::forward<Args>(args)...);
    }
    _llk_math_eltwise_unary_sfpu_done_();
}

// Split-dest overload: source tile index (dst_index_in) is used to position
// the dest face pointer at the start; the callable then receives both indices
// so an SFPU op can read from dst_index_in and write to dst_index_out.
// The dst-bound assert lives in ckernel::_sfpu_check_and_call_ (see the
// per-arch llk_math_eltwise_unary_sfpu_macros.h), so it is intentionally not
// duplicated here.
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index_in, std::uint32_t dst_index_out, int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args)
{
    _llk_math_eltwise_unary_sfpu_start_(dst_index_in);

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::R)
    {
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    }
    else if (mode == VectorMode::C)
    {
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else if (mode == VectorMode::RC)
    {
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
    }
    _llk_math_eltwise_unary_sfpu_done_();
}
