// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
using namespace ckernel;
using namespace ckernel::trisc;

// Thread-agnostic semaphore primitives for LLK-level producer/consumer
// handshakes. Each helper is a thin wrapper over the underlying TT instruction
// (SEMINIT / SEMWAIT / SEMGET / SEMPOST) with a single semaphore-index argument
// in the 0..31 range; the producer/consumer role is assigned at the call site.
//
// Usage notes:
//  - `sem_index` is the integer semaphore id (e.g. UNPACK_TO_DEST_UNPACK_SEMAPHORE = 4),
//    NOT the one-hot bitmask form (e.g. p_stall::SEMAPHORE_4 = 0x10).
//  - `_llk_sync_wait_` blocks until the semaphore satisfies the given condition
//    (typically p_stall::STALL_ON_ZERO or p_stall::STALL_ON_MAX). The stall
//    resource (e.g. STALL_UNPACK, STALL_TDMA) is a template parameter.
//  - `_llk_sync_get_` and `_llk_sync_post_` accept an optional pre-stall
//    resource via the template parameter, matching `t6_semaphore_get/post`.

// Initialize semaphore `sem_index` with the given max value and initial value.
inline void _llk_sync_init_(std::uint8_t sem_index, std::uint32_t max, std::uint32_t init)
{
    TTI_SEMINIT(max, init, 0, semaphore::t6_sem(sem_index));
}

// Block on semaphore `sem_index` until it satisfies `condition`
// (p_stall::STALL_ON_ZERO or p_stall::STALL_ON_MAX). `StallRes` is the resource
// the calling thread holds while blocked (e.g. p_stall::STALL_UNPACK).
template <std::uint32_t StallRes>
inline void _llk_sync_wait_(std::uint8_t sem_index, std::uint32_t condition)
{
    TTI_SEMWAIT(StallRes, condition, 0, semaphore::t6_sem(sem_index));
}

// Decrement semaphore `sem_index`. Optionally stall on `WaitRes` first.
template <std::uint32_t WaitRes = p_stall::NOTHING>
inline void _llk_sync_get_(std::uint8_t sem_index)
{
    t6_semaphore_get<WaitRes>(sem_index);
}

// Increment semaphore `sem_index`. Optionally stall on `WaitRes` first.
template <std::uint32_t WaitRes = p_stall::NOTHING>
inline void _llk_sync_post_(std::uint8_t sem_index)
{
    t6_semaphore_post<WaitRes>(sem_index);
}
