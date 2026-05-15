// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU MAX along H or W (Int32/Float32). MIN is dispatched to reduce_sfpu_{h,w}_neg.cpp.
// Routes to the SFPU branch in compute_kernel_lib::reduce<> via the REDUCE_FORMAT template arg.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE,
        REDUCE_FORMAT>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        // sfpu_reduce ignores the scaler CB (LLK takes no scalar), so the user scalar is applied
        // here per output tile. The host requests reduce with scaler=1.0 and packs the user scalar
        // bits as a compile-time arg.
        [](uint32_t dst_idx) {
            constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
            compute_kernel_lib::detail::sfpu_post_mul_tile<REDUCE_FORMAT>(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );
}
