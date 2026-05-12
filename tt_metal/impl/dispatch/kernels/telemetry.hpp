// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "internal/risc_attribs.h"

#include <cstdint>

struct NoTelemetry {};

template <typename Telemetry, uint32_t telemetry_addr>
FORCE_INLINE volatile tt_l1_ptr Telemetry* get_telemetry_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr Telemetry*>(telemetry_addr);
}

template <typename Telemetry, bool enabled>
class TelemetryBlockGuardImpl;

template <typename Telemetry>
class TelemetryBlockGuardImpl<Telemetry, true> {
public:
    FORCE_INLINE explicit TelemetryBlockGuardImpl(volatile tt_l1_ptr Telemetry* telemetry) : telemetry_(telemetry) {}

    FORCE_INLINE ~TelemetryBlockGuardImpl() {
        if (blocked_) {
            telemetry_->unblocked_count++;
        }
    }

    TelemetryBlockGuardImpl(const TelemetryBlockGuardImpl&) = delete;
    TelemetryBlockGuardImpl& operator=(const TelemetryBlockGuardImpl&) = delete;
    TelemetryBlockGuardImpl(TelemetryBlockGuardImpl&& other) = delete;
    TelemetryBlockGuardImpl& operator=(TelemetryBlockGuardImpl&& other) = delete;

    FORCE_INLINE void mark_blocked() {
        if (!blocked_) {
            telemetry_->blocked_count++;
            blocked_ = true;
        }
    }

private:
    volatile tt_l1_ptr Telemetry* telemetry_;
    bool blocked_{false};
};

template <typename Telemetry>
class TelemetryBlockGuardImpl<Telemetry, false> {
public:
    FORCE_INLINE explicit TelemetryBlockGuardImpl(volatile tt_l1_ptr Telemetry*) {}
    FORCE_INLINE ~TelemetryBlockGuardImpl() = default;
    FORCE_INLINE void mark_blocked() {}
};

template <typename Telemetry, bool enabled>
using TelemetryBlockGuard = TelemetryBlockGuardImpl<Telemetry, enabled>;
