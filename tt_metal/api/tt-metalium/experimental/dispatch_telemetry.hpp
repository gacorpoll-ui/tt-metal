// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

/**
 * @brief Magic constant to verify data is valid.
 */
constexpr uint32_t MAGIC_CONSTANT = 0x12345678;

/**
 * @brief Telemetry for prefetch.
 */
struct PrefetchTelemetry {
    const uint32_t version = 1;
    const uint32_t magic_constant = MAGIC_CONSTANT;
    uint64_t blocked_count = 0;
    uint64_t unblocked_count = 0;
    uint64_t command_count = 0;
};

/**
 * @brief Telemetry for dispatch.
 */
struct DispatchTelemetry {
    const uint32_t version = 1;
    const uint32_t magic_constant = MAGIC_CONSTANT;
    uint64_t blocked_count = 0;
    uint64_t unblocked_count = 0;
};

}  // namespace tt::tt_metal
