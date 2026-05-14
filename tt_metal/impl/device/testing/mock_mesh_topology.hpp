// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FIX CR (#42429): Mock mesh topology for testing FIX CN/CO/CP without hardware.
//
// Provides a pure-data MockMesh that represents a multi-device topology
// with ETH connections, MMIO/non-MMIO classification, bc_deadlock channel
// sets, and relay-broken state.  Test helper functions mirror the exact
// logic of FIX CN, FIX CO, and FIX CP against this mock — no UMD, no
// Cluster, no real devices.

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tt::tt_metal::testing {

// ChipId matches tt::ChipId = int throughout tt-metal.
using MockChipId = int;

struct MockDevice {
    MockChipId chip_id;
    bool is_mmio = false;
    bool relay_broken = false;
    std::unordered_set<uint32_t> bc_deadlock_channels;
    // ETH channel -> (peer_chip_id, peer_chan_id)
    std::unordered_map<uint32_t, std::pair<MockChipId, uint32_t>> eth_connections;
};

struct MockMesh {
    std::vector<MockDevice> devices;

    const MockDevice* get_device(MockChipId id) const {
        for (const auto& d : devices) {
            if (d.chip_id == id) return &d;
        }
        return nullptr;
    }

    bool is_mmio(MockChipId id) const {
        const auto* d = get_device(id);
        return d && d->is_mmio;
    }

    bool is_relay_broken(MockChipId id) const {
        const auto* d = get_device(id);
        return d && d->relay_broken;
    }

    std::unordered_set<uint32_t> get_bc_deadlock_channels(MockChipId id) const {
        const auto* d = get_device(id);
        return d ? d->bc_deadlock_channels : std::unordered_set<uint32_t>{};
    }

    // Returns {peer_chip_id, peer_chan} for a given (chip, chan) or nullopt.
    std::optional<std::pair<MockChipId, uint32_t>> get_eth_peer(MockChipId chip, uint32_t chan) const {
        const auto* d = get_device(chip);
        if (!d) return std::nullopt;
        auto it = d->eth_connections.find(chan);
        if (it == d->eth_connections.end()) return std::nullopt;
        return it->second;
    }

    // Convenience: set of all MMIO chip IDs.
    std::unordered_set<MockChipId> mmio_ids_set() const {
        std::unordered_set<MockChipId> s;
        for (const auto& d : devices) {
            if (d.is_mmio) s.insert(d.chip_id);
        }
        return s;
    }
};

// ─── Key packing (mirrors risc_firmware_initializer.cpp make_mmio_skip_key) ───

inline uint64_t make_mmio_skip_key(MockChipId chip_id, uint32_t chan) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(chip_id)) << 32) | static_cast<uint64_t>(chan);
}

// ─── FIX CN predicate ─────────────────────────────────────────────────────────
// Returns true if the channel should be skipped in fabric_firmware_init::teardown
// (i.e. it is a bc_deadlock channel on this device).
// Mirrors: fabric_firmware_initializer.cpp ~line 1103.

inline bool should_skip_cn(MockChipId chip_id, uint32_t eth_chan_id, const MockMesh& mesh) {
    return mesh.get_bc_deadlock_channels(chip_id).count(eth_chan_id) > 0;
}

// ─── FIX CO logic ─────────────────────────────────────────────────────────────
// For every bc_deadlock channel on a relay-broken non-MMIO device, find its MMIO
// ETH peer and add that peer to the mmio_ac_skip_channels map.
// Mirrors: risc_firmware_initializer.cpp ~line 546-601.

inline std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>> compute_mmio_ac_skip_co(
    const MockMesh& mesh,
    const std::unordered_set<MockChipId>& relay_broken_non_mmio) {

    std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>> mmio_ac_skip_channels;
    const auto mmio_ids = mesh.mmio_ids_set();

    for (const MockChipId non_mmio_id : relay_broken_non_mmio) {
        const MockDevice* dev = mesh.get_device(non_mmio_id);
        if (!dev) continue;

        const auto& bcd = dev->bc_deadlock_channels;
        if (bcd.empty()) continue;

        for (const uint32_t bc_chan : bcd) {
            auto peer_opt = mesh.get_eth_peer(non_mmio_id, bc_chan);
            if (!peer_opt) continue;

            const auto& [peer_chip_id, peer_chan] = *peer_opt;
            // Only skip if the peer is an MMIO device.
            if (!mmio_ids.count(peer_chip_id)) continue;

            const uint64_t skip_key = make_mmio_skip_key(peer_chip_id, peer_chan);
            mmio_ac_skip_channels.emplace(skip_key, std::make_pair(non_mmio_id, bc_chan));
        }
    }
    return mmio_ac_skip_channels;
}

// ─── FIX CP logic ─────────────────────────────────────────────────────────────
// When FIX CA fails for a device, adds ALL MMIO ETH peers of ALL channels on
// that device to mmio_ac_skip_channels.
// Mirrors: risc_firmware_initializer.cpp ~line 825-870.

inline void extend_mmio_ac_skip_cp(
    const MockMesh& mesh,
    const std::unordered_set<MockChipId>& ca_failed_devices,
    std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>>& mmio_ac_skip_channels) {

    const auto mmio_ids = mesh.mmio_ids_set();

    for (const MockChipId failed_non_mmio_id : ca_failed_devices) {
        const MockDevice* dev = mesh.get_device(failed_non_mmio_id);
        if (!dev) continue;

        for (const auto& [eth_chan, peer_info] : dev->eth_connections) {
            const auto& [peer_chip_id, peer_chan] = peer_info;
            if (!mmio_ids.count(peer_chip_id)) continue;

            const uint64_t skip_key = make_mmio_skip_key(peer_chip_id, peer_chan);
            mmio_ac_skip_channels.emplace(skip_key, std::make_pair(failed_non_mmio_id, eth_chan));
        }
    }
}

}  // namespace tt::tt_metal::testing
