// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh_dest.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#else
#include "experimental/dataflow_buffer.h"
#endif

void kernel_main() {
    uint32_t NHtWt = get_compile_time_arg_val(0);

#ifndef ARCH_QUASAR
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    constexpr uint32_t dfb_in_id = get_compile_time_arg_val(1);
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(2);
    experimental::DataflowBuffer dfb_in(dfb_in_id);
    experimental::DataflowBuffer dfb_out(dfb_out_id);
    unary_op_init_common(dfb_in.get_id(), dfb_out.get_id());
#endif

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh_dest each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
#ifndef ARCH_QUASAR
        cb0.wait_front(1);
        cb16.reserve_back(1);

        tile_regs_acquire();
        copy_tile_init(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        transpose_wh_dest_init_short<DST_ACCUM_MODE>();
        transpose_wh_dest<DST_ACCUM_MODE>(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();

        cb16.push_back(1);
        cb0.pop_front(1);
#else
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        copy_tile_init(dfb_in.get_id());
        copy_tile(dfb_in.get_id(), 0, 0);

        transpose_wh_dest_init_short<DST_ACCUM_MODE>();
        transpose_wh_dest<DST_ACCUM_MODE>(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb_out.get_id());
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
#endif
    }
}
