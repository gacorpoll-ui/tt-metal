// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FIX CK (#42429): Lightweight fault injection for hardware-timing-dependent test paths.
//
// Provides injectable failure points for:
//   H2 — relay write failure (FIX CA / FIX AY dual failure → FIX CG)
//   H3 — flush timeout in write_core_immediate (FIX CH elapsed-ms log)
//
// All state is stored in a static struct guarded by TT_ENABLE_FAULT_INJECTION.
// In production builds (no define), every hook compiles to nothing.
// State is NOT thread-local — callers must call reset() between tests.

#pragma once

#ifdef TT_ENABLE_FAULT_INJECTION

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tt::tt_metal::testing {

// Compound key: (chip_id, core_x, core_y).  core_x/core_y packed into uint64_t.
// For relay write failures we key on chip_id alone (all cores on that chip fail).
// For flush timeout we key on chip_id alone.

struct FaultInjector {
    // ── H2: relay write failure ──────────────────────────────────────────────
    // Arm: any call to assert_risc_reset_at_core_write_only or
    // deassert_risc_reset_at_core_write_only for this chip will throw.
    static void arm_relay_write_failure(int chip_id) {
        std::lock_guard<std::mutex> lk(mu_);
        relay_fail_chips_.insert(chip_id);
    }

    static bool should_fail_relay_write(int chip_id) {
        std::lock_guard<std::mutex> lk(mu_);
        return relay_fail_chips_.count(chip_id) > 0;
    }

    // ── H3: flush timeout ────────────────────────────────────────────────────
    // Arm: write_core_immediate for this chip will throw a flush-timeout
    // exception after simulating `elapsed_ms` of delay.
    static void arm_flush_timeout(int chip_id, uint32_t elapsed_ms) {
        std::lock_guard<std::mutex> lk(mu_);
        flush_timeout_chips_[chip_id] = elapsed_ms;
    }

    static bool get_flush_timeout(int chip_id, uint32_t* out_elapsed_ms) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = flush_timeout_chips_.find(chip_id);
        if (it == flush_timeout_chips_.end()) return false;
        if (out_elapsed_ms) *out_elapsed_ms = it->second;
        return true;
    }

    // ── Reset all state between tests ────────────────────────────────────────
    static void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        relay_fail_chips_.clear();
        flush_timeout_chips_.clear();
    }

private:
    static inline std::mutex mu_;
    static inline std::unordered_set<int> relay_fail_chips_;
    static inline std::unordered_map<int, uint32_t> flush_timeout_chips_;
};

}  // namespace tt::tt_metal::testing

#endif  // TT_ENABLE_FAULT_INJECTION
