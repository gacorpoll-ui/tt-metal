// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

/* TODO: remove notes
### Prefetch
* Counter for host unblocked -> blocked transition
* Counter for host blocked -> unblocked transition
* Command counter

### Dispatch
* Counter for host unblocked -> blocked transition
* Counter for host blocked -> unblocked transition

### Dispatch_s (Or dispatch if no dispatch_s)
* timestamp of the last time it sent a go message to launch a program
* accumulated time while program is running
* current time
*/

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

/**
 * @brief Telemetry for dispatch_s (or dispatch if there is no dispatch_s running)
 */
struct DispatchSTelemetry {
    const uint32_t version = 1;
    const uint32_t magic_constant = MAGIC_CONSTANT;
    uint64_t last_go_message_cycle = 0;
    // TODO: do these in a follow up PR.
    uint64_t accumulated_program_cycles = 0; // mainly used by separate kernel that will poll worker core count
    uint64_t current_cycle = 0;
};

}  // namespace tt::tt_metal
