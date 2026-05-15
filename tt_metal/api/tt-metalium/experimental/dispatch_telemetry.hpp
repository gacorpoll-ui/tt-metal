// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

// TODO: Split api between types and functions to support kernel and firmware builds.
#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
#include <optional>

#include "core_coord.hpp"
#endif

namespace tt::tt_metal {

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
class IDevice;
#endif

/**
 * @brief Expected signature for validating that a telemetry buffer contains dispatch telemetry data.
 */
constexpr uint32_t DISPATCH_TELEMETRY_SIGNATURE = 0x12345678;
constexpr uint32_t DISPATCH_TELEMETRY_VERSION = 1;

/**
 * @brief Telemetry for prefetch.
 */
struct PrefetchTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = DISPATCH_TELEMETRY_SIGNATURE;
    uint64_t blocked_by_host_count = 0;
    uint64_t unblocked_by_host_count = 0;
    uint64_t command_count = 0;
};

/**
 * @brief Telemetry for dispatch.
 */
struct DispatchTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = DISPATCH_TELEMETRY_SIGNATURE;
    uint64_t blocked_by_host_count = 0;
    uint64_t unblocked_by_host_count = 0;
};

// Used to determine the size of the L1 buffer that dispatch_mem_map allocates
// Note: If new telemetry types are added, update this calculation
constexpr size_t DISPATCH_TELEMETRY_SIZE = std::max(
    sizeof(DispatchTelemetry),
    sizeof(PrefetchTelemetry));

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
/**
 * @brief Read the DispatchTelemetry block from a dispatch core's L1.
 *
 * @param device                Device that owns the dispatch core.
 * @param dispatch_logical_core Logical coord of the dispatch core to sample.
 * @param core_type             CoreType of the dispatch core (WORKER for tensix-based dispatch,
 *                              ETH for ethernet-based dispatch).
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<DispatchTelemetry> read_dispatch_telemetry(
    IDevice* device, const CoreCoord& dispatch_logical_core, CoreType core_type = CoreType::WORKER);

/**
 * @brief Read the PrefetchTelemetry block from a prefetch core's L1.
 *
 * @param device                Device that owns the prefetch core.
 * @param prefetch_logical_core Logical coord of the prefetch core to sample.
 * @param core_type             CoreType of the prefetch core (WORKER for tensix-based prefetch,
 *                              ETH for ethernet-based prefetch).
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<PrefetchTelemetry> read_prefetch_telemetry(
    IDevice* device, const CoreCoord& prefetch_logical_core, CoreType core_type = CoreType::WORKER);
#endif  // !KERNEL_BUILD && !FW_BUILD

}  // namespace tt::tt_metal
