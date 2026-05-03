// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_sync.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_common_api.h"
#include "experimental/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

/**
 *
 * @brief Initialize selected unpacker to unpack a single tile
 *
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, supported for SrcA and SrcB
 * @tparam IS_32b_DEST_EN: Enable using Math destination Register in 32-bit mode
 * @param operand: The input operand circular buffer
 *
 * This function initializes unpacker0 to unpack a single tile
 * from the input circular buffer to srcA/dest register.
 */
template <bool TRANSPOSE_EN, bool IS_32b_DEST_EN>
inline void llk_unpack_A_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    // Runtime format check: 32-bit dst format routes through unpack-to-dest.
    // Math thread sits out while unpacker writes to dest directly, the packer
    // reads it via a dedicated sem-4/7 handshake (initialized below).
    const std::uint32_t dst_format = unpack_dst_format[operand_id];
    if (dst_format == (std::uint32_t)DataFormat::Float32 ||
        dst_format == (std::uint32_t)DataFormat::Int32) {
        // Program ALU implied-format register (FP32 vs Int32 from src format).
        if (static_cast<DataFormat>(unpack_src_format[operand_id]) == DataFormat::Int32) {
            _llk_unpack_to_dest_hw_configure_<false /*EN_FP32*/, true /*EN_INT32*/>();
        } else {
            _llk_unpack_to_dest_hw_configure_<true /*EN_FP32*/, false /*EN_INT32*/>();
        }
        // Unpacker MOP for UNP_DEST 
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false /*TRANSPOSE_EN*/, IS_32b_DEST_EN>(operand_id);
        
        // Init the two-semaphore unpack <-> pack handshake.
        // Sem UNPACK_TO_DEST_UNPACK ("filled banks"): max=N, init=0, empty until unpacker posts.
        // Sem UNPACK_TO_DEST_PACK   ("free banks"):   max=N, init=N, all banks free at start.
        constexpr std::uint32_t N = (DST_SYNC_MODE == DstSync::SyncFull) ? 1 : 2;
        _llk_sync_init_(UNPACK_TO_DEST_UNPACK_SEMAPHORE, N, 0);
        _llk_sync_init_(UNPACK_TO_DEST_PACK_SEMAPHORE,   N, N);
        if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
            _reset_dest_register_offset_();
            _set_dest_section_base_<ckernel::unpack::TRISC_ID>(_get_dest_buffer_base_());
        }
        return;
    }

    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32b_DEST_EN>(operand_id);
}

/**
 *
 * @brief Initialize unpacker0 with dest reuse support
 *
 * Overload matching Blackhole/Wormhole API signature to support binary dest reuse operations.
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    [[maybe_unused]] const std::uint32_t transpose_of_faces = 0,
    [[maybe_unused]] const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
    static_assert(BType == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");

    // TODO (tt-metal #42916): Once runtime asserts are added, add asserts for unsupported features above and for valid
    // transpose_of_faces and within_face_16x16_transpose values

    // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
    _llk_unpack_unary_operand_init_<
        p_unpacr::UNP_A,
        false /* TRANSPOSE_EN */,
        false /* IS_32b_DEST_EN */,
        binary_reuse_dest>(operand_id);
}

/**
 *
 * @brief Unpacks a single operand, unpacker0 is used
 *
 * @param operand: The logical dataflow buffer id
 * @param tile_index: The index in the input CB to read from
 *
 * This function unpacks a single operand from the input circular buffer to srcA/dest register.
 */
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    // Number of tiles the read pointer has advanced from DFB base
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t l1_tile_index =
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx + tile_index;


    const std::uint32_t dst_format = unpack_dst_format[operand_id];
    if (dst_format == (std::uint32_t)DataFormat::Float32 ||
        dst_format == (std::uint32_t)DataFormat::Int32) {
        
        // Block until at least one dest bank is free (PACK sem > 0), then claim it.
        _llk_sync_wait_<p_stall::STALL_UNPACK>(UNPACK_TO_DEST_PACK_SEMAPHORE, p_stall::STALL_ON_ZERO);
        _llk_sync_get_(UNPACK_TO_DEST_PACK_SEMAPHORE);
      
        _llk_unpack_unary_operand_<p_unpacr::UNP_DEST>(l1_tile_index);

        // Drain UNPACK0 before posting "tile filled" or the post races ahead of the writes.
        _llk_sync_post_<p_stall::UNPACK0>(UNPACK_TO_DEST_UNPACK_SEMAPHORE);
        if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
            _update_dest_register_offset_<true /*EN_32BIT_DEST*/>();
            const std::uint32_t base_addr = _get_dest_buffer_base_();
            TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::UNPACK0);
            _set_dest_section_base_<ckernel::unpack::TRISC_ID>(base_addr);
        }
        return;
    }

    WAYPOINT("UPAW");
    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(l1_tile_index);
    WAYPOINT("UPAD");
}

/**
 *
 * @brief Unpacks a single operand with dest reuse support
 *
 * Overload matching Blackhole/Wormhole API signature to support binary dest reuse operations.
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t l1_tile_index =
        g_dfb_interface[operand_id].tc_slots[g_dfb_interface[operand_id].tc_idx].rd_entry_idx + tile_index;

    static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
    static_assert(BType == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");

    WAYPOINT("UPAW");
    // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
    _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest>(l1_tile_index);
    WAYPOINT("UPAD");
}

/**
 * @brief Unpacks a contiguous block of tiles with unpacker0.
 *
 * @param operand The logical dataflow buffer id.
 * @param start_tile_index The starting tile index within the input buffer.
 * @param ntiles The number of consecutive tiles to unpack.
 *
 * The tiles are read from the operand buffer starting at start_tile_index
 * and unpacked into srcA one tile at a time.
 */
// TODO: AM; Optimize block calls by using ntiles per unpack, issue #40798
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    std::uint32_t l1_tile_index =
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx + start_tile_index;

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(l1_tile_index);
        l1_tile_index += 1;
        WAYPOINT("UPAD");
    }
}
