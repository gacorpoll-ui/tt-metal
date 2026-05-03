// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h" 
#include "cunpack_common.h"
using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief Programs unpacker l1 info & source register format
 * @tparam UNP_SEL: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src: Contains source reg format
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_hw_configure_(const tdma_descriptor_t& tdma_desc_src)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_S) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_S/UNP_DEST");

    // RT: make defines to aggregate the source format address, to make the below a single function
    // Program src formats
    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_S)
    {
        cfg_rmw(THCON_UNPACKER2_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
}

// RT: make defines to aggregate _llk_unpack_hw_configure_ calls into one
/**
 * @brief Programs unpacker l1 info & source register format for unary operation
 * @tparam UNP_SEL: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src: Contains L1 buffer descriptor information & source reg format for Src Reg
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_configure_unary_(const tdma_descriptor_t& tdma_desc_src)
{
    _llk_unpack_hw_configure_<UNP_SEL>(tdma_desc_src);
}

/**
 * @brief Programs unpacker l1 info & source register format for binary operation
 * @tparam UNP_SEL0/1: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src0/1: Contains L1 buffer descriptor information & source reg format for Src Reg
 */
template <std::uint32_t UNP_SEL_0, std::uint32_t UNP_SEL_1>
inline void _llk_unpack_configure_binary_(const tdma_descriptor_t& tdma_desc_src0, const tdma_descriptor_t& tdma_desc_src1)
{
    _llk_unpack_hw_configure_<UNP_SEL_0>(tdma_desc_src0);
    _llk_unpack_hw_configure_<UNP_SEL_1>(tdma_desc_src1);
}

/**
 * @brief Programs the global ALU format-spec register from the unpack thread for
 *        the unpack-to-dest path.
 *
 * In unpack-to-dest mode the math thread sits idle: the unpacker writes dest directly via UNPACR with UNP_DEST and the packer reads it through the
 * sem-4/7 handshake. The ALU_FORMAT_SPEC_REG (cfg addr 0..2) is global and controls dest interpretation specifically `ALU_ACC_CTRL_*_enabled` selects
 * 32-bit dest mode (4-byte entries) vs the default 16-bit mode (2-byte entries).
 *
 * @tparam EN_FP32_MATH_FORMAT  Set to true for Float32 dest mode.
 * @tparam EN_INT32_MATH_FORMAT Set to true for Int32 dest mode.
 *                              Both false ⇒ default 16-bit (Float16/Float16_b).
 *                              Mutually exclusive: at most one should be true.
 */
template <bool EN_FP32_MATH_FORMAT, bool EN_INT32_MATH_FORMAT>
inline void _llk_unpack_to_dest_hw_configure_()
{
    static_assert(!(EN_FP32_MATH_FORMAT && EN_INT32_MATH_FORMAT),
                  "Float32 and Int32 dest modes are mutually exclusive");

    ckernel::math::alu_config_u alu_config;
    for (std::uint32_t i = 0; i < ckernel::math::NUM_WORDS_ALU_FORMAT; i++)
    {
        alu_config.val[i] = 0;
    }

    alu_config.f.ALU_ACC_CTRL_Fp32_enabled      = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_INT8_math_enabled = EN_INT32_MATH_FORMAT;

    for (std::uint32_t i = 0; i < ckernel::math::NUM_WORDS_ALU_FORMAT; i++)
    {
        cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 + i] = alu_config.val[i];
    }
}

template <DstSync DST>
inline void _llk_unpack_dest_dvalid_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, p_stall::UNPACK0);
    TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::UNPACK_TO_DEST, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        // For DstSync::SyncFull issue a CLEARDVALID instruction for dest bank1 as well in order to use full dest register
        // Reset dest bank id to 0 for the given dest client to ensure SyncFull starts from bank0
        TTI_CLEARDVALID(0, 0, 0, p_cleardvalid::UNPACK_TO_DEST, p_cleardvalid::UNPACK_TO_DEST, 0);
    }
}

/**
 * @brief Sets dummy SrcB dvalid
 */
inline void _llk_unpack_set_srcB_dummy_valid_()
{
    TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Set_Dvalid*/, 0, 0, 0, p_unpacr::UNP_NOP);
}
