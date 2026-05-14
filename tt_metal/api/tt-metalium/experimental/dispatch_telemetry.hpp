// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <algorithm>

namespace tt::tt_metal {

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

// TODO: fix
DispatchTelemetry read_dispatch_telemetry(
    IDevice* device, const CoreCoord& dispatch_logical_core, CoreType core_type = CoreType::WORKER) {
    // Telemetry lives at a fixed dispatch-core-local L1 offset assigned by DispatchMemMap.
    const auto& dispatch_mem_map = MetalContext::instance().dispatch_mem_map();
    uint32_t addr = dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_TELEMETRY);

    // read_core needs a virtual (noc-addressable) coord, not the logical one the caller has.
    CoreCoord virtual_core = device->virtual_core_from_logical_core(dispatch_logical_core, core_type);

    // Make sure any in-flight kernel writes to L1 are visible before we sample.
    tt::Cluster::instance().l1_barrier(device->id());

    DispatchTelemetry out{};
    tt::Cluster::instance().read_core(
        &out, sizeof(out), tt_cxy_pair(device->id(), virtual_core), addr);

    // Sanity-check the buffer actually contains a current-version DispatchTelemetry.
    TT_FATAL(
        out.signature == DISPATCH_TELEMETRY_SIGNATURE,
        "DispatchTelemetry signature mismatch on chip {} core ({},{}) @ 0x{:x}: got 0x{:x}",
        device->id(),
        dispatch_logical_core.x,
        dispatch_logical_core.y,
        addr,
        out.signature);
    TT_FATAL(
        out.version == DISPATCH_TELEMETRY_VERSION,
        "DispatchTelemetry version mismatch on chip {}: got {}, expected {}",
        device->id(),
        out.version,
        DISPATCH_TELEMETRY_VERSION);
    return out;
}

}  // namespace tt::tt_metal
