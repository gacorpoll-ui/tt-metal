// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FIX CR (#42429): Mock-based unit tests for FIX CN, FIX CO, and FIX CP.
//
// These tests exercise the bc_deadlock skip logic, MMIO AC skip channel
// population (relay-broken non-MMIO + FIX CA failure), and the FIX CN
// predicate against a pure-data MockMesh.  No hardware required.
//
// See tt_metal/impl/device/testing/mock_mesh_topology.hpp for the mock
// infrastructure and extracted helper functions.

#include <gtest/gtest.h>

#include "impl/device/testing/mock_mesh_topology.hpp"

namespace tt::tt_metal::testing::test {

// ═══════════════════════════════════════════════════════════════════════════════
// Topology builders — small helpers that construct common MockMesh configs
// ═══════════════════════════════════════════════════════════════════════════════

// Minimal T3K-ish topology:
//   Device 0 (MMIO)      chan 4 <-> Device 4 (non-MMIO) chan 0
//   Device 0 (MMIO)      chan 5 <-> Device 4 (non-MMIO) chan 1
//   Device 0 (MMIO)      chan 6 <-> Device 4 (non-MMIO) chan 2
//   Device 0 (MMIO)      chan 7 <-> Device 4 (non-MMIO) chan 3
//   Device 1 (MMIO)      chan 4 <-> Device 5 (non-MMIO) chan 0
//   Device 1 (MMIO)      chan 5 <-> Device 5 (non-MMIO) chan 1
static MockMesh make_t3k_topology() {
    MockMesh mesh;

    MockDevice dev0;
    dev0.chip_id = 0;
    dev0.is_mmio = true;
    dev0.eth_connections = {
        {4, {4, 0}},
        {5, {4, 1}},
        {6, {4, 2}},
        {7, {4, 3}},
    };

    MockDevice dev1;
    dev1.chip_id = 1;
    dev1.is_mmio = true;
    dev1.eth_connections = {
        {4, {5, 0}},
        {5, {5, 1}},
    };

    MockDevice dev4;
    dev4.chip_id = 4;
    dev4.is_mmio = false;
    dev4.eth_connections = {
        {0, {0, 4}},
        {1, {0, 5}},
        {2, {0, 6}},
        {3, {0, 7}},
    };

    MockDevice dev5;
    dev5.chip_id = 5;
    dev5.is_mmio = false;
    dev5.eth_connections = {
        {0, {1, 4}},
        {1, {1, 5}},
    };

    mesh.devices = {dev0, dev1, dev4, dev5};
    return mesh;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIX CN tests — bc_deadlock channel skip predicate
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FixCNMockTest, BcDeadlockChannelSkipped) {
    MockMesh mesh;
    MockDevice dev0;
    dev0.chip_id = 0;
    dev0.is_mmio = true;
    dev0.bc_deadlock_channels = {3};
    mesh.devices = {dev0};

    EXPECT_TRUE(should_skip_cn(0, 3, mesh));
    EXPECT_FALSE(should_skip_cn(0, 4, mesh));
}

TEST(FixCNMockTest, NonDeadlockChannelNotSkipped) {
    MockMesh mesh;
    MockDevice dev0;
    dev0.chip_id = 0;
    dev0.is_mmio = true;
    // No bc_deadlock channels
    mesh.devices = {dev0};

    EXPECT_FALSE(should_skip_cn(0, 0, mesh));
    EXPECT_FALSE(should_skip_cn(0, 3, mesh));
    EXPECT_FALSE(should_skip_cn(0, 15, mesh));
}

TEST(FixCNMockTest, MultipleDeadlockChannels) {
    MockMesh mesh;
    MockDevice dev0;
    dev0.chip_id = 0;
    dev0.is_mmio = true;
    dev0.bc_deadlock_channels = {6, 7, 8, 9};
    mesh.devices = {dev0};

    // All four deadlock channels skipped
    EXPECT_TRUE(should_skip_cn(0, 6, mesh));
    EXPECT_TRUE(should_skip_cn(0, 7, mesh));
    EXPECT_TRUE(should_skip_cn(0, 8, mesh));
    EXPECT_TRUE(should_skip_cn(0, 9, mesh));

    // Other channels not skipped
    EXPECT_FALSE(should_skip_cn(0, 0, mesh));
    EXPECT_FALSE(should_skip_cn(0, 5, mesh));
    EXPECT_FALSE(should_skip_cn(0, 10, mesh));
}

TEST(FixCNMockTest, UnknownDeviceNotSkipped) {
    MockMesh mesh;
    // Empty mesh — device 99 does not exist.
    EXPECT_FALSE(should_skip_cn(99, 0, mesh));
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIX CO tests — bc_deadlock on relay-broken non-MMIO → MMIO AC skip
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FixCOMockTest, BcDeadlockChanOnRelayBrokenNonMMIO_PopulatesSkip) {
    // Device 0 (MMIO) chan 4 <-> Device 4 (non-MMIO) chan 0
    // Device 4: bc_deadlock={0}, relay_broken=true
    auto mesh = make_t3k_topology();
    auto* dev4 = const_cast<MockDevice*>(mesh.get_device(4));
    dev4->relay_broken = true;
    dev4->bc_deadlock_channels = {0};

    std::unordered_set<MockChipId> relay_broken_non_mmio = {4};
    auto skip = compute_mmio_ac_skip_co(mesh, relay_broken_non_mmio);

    // Expected: MMIO device 0, chan 4 is in the skip set.
    uint64_t expected_key = make_mmio_skip_key(0, 4);
    ASSERT_EQ(skip.size(), 1u);
    ASSERT_TRUE(skip.count(expected_key));
    // The value records the non-MMIO peer that caused the skip.
    EXPECT_EQ(skip[expected_key].first, 4);   // non_mmio_id
    EXPECT_EQ(skip[expected_key].second, 0u);  // bc_chan
}

TEST(FixCOMockTest, RelayBrokenButNoBcDeadlock_NoSkip) {
    auto mesh = make_t3k_topology();
    auto* dev4 = const_cast<MockDevice*>(mesh.get_device(4));
    dev4->relay_broken = true;
    // bc_deadlock_channels is empty (default)

    std::unordered_set<MockChipId> relay_broken_non_mmio = {4};
    auto skip = compute_mmio_ac_skip_co(mesh, relay_broken_non_mmio);

    EXPECT_TRUE(skip.empty());
}

TEST(FixCOMockTest, BcDeadlockButNotInRelayBrokenSet_NoSkip) {
    // Device 4 has bc_deadlock channels but is NOT in relay_broken_non_mmio.
    // (In real code, relay_broken_non_mmio is the iteration set, so if device
    // 4 isn't in it, FIX CO never looks at device 4.)
    auto mesh = make_t3k_topology();
    auto* dev4 = const_cast<MockDevice*>(mesh.get_device(4));
    dev4->bc_deadlock_channels = {0};

    std::unordered_set<MockChipId> relay_broken_non_mmio;  // empty
    auto skip = compute_mmio_ac_skip_co(mesh, relay_broken_non_mmio);

    EXPECT_TRUE(skip.empty());
}

TEST(FixCOMockTest, MultipleNonMMIODevices_AllSkipsPropagated) {
    auto mesh = make_t3k_topology();

    // Device 4: bc_deadlock on chan 0 and 1
    auto* dev4 = const_cast<MockDevice*>(mesh.get_device(4));
    dev4->relay_broken = true;
    dev4->bc_deadlock_channels = {0, 1};

    // Device 5: bc_deadlock on chan 0
    auto* dev5 = const_cast<MockDevice*>(mesh.get_device(5));
    dev5->relay_broken = true;
    dev5->bc_deadlock_channels = {0};

    std::unordered_set<MockChipId> relay_broken_non_mmio = {4, 5};
    auto skip = compute_mmio_ac_skip_co(mesh, relay_broken_non_mmio);

    // Device 4 chan 0 -> MMIO dev 0 chan 4
    // Device 4 chan 1 -> MMIO dev 0 chan 5
    // Device 5 chan 0 -> MMIO dev 1 chan 4
    EXPECT_EQ(skip.size(), 3u);
    EXPECT_TRUE(skip.count(make_mmio_skip_key(0, 4)));
    EXPECT_TRUE(skip.count(make_mmio_skip_key(0, 5)));
    EXPECT_TRUE(skip.count(make_mmio_skip_key(1, 4)));
}

TEST(FixCOMockTest, BcDeadlockChanWithNonMMIOPeer_NotAdded) {
    // If the bc_deadlock channel's peer is ALSO non-MMIO, it should NOT be
    // added to mmio_ac_skip_channels (FIX AC only resets MMIO channels).
    MockMesh mesh;

    MockDevice dev4;
    dev4.chip_id = 4;
    dev4.is_mmio = false;
    dev4.relay_broken = true;
    dev4.bc_deadlock_channels = {0};
    dev4.eth_connections = {{0, {5, 0}}};  // Peer is device 5, also non-MMIO

    MockDevice dev5;
    dev5.chip_id = 5;
    dev5.is_mmio = false;
    dev5.eth_connections = {{0, {4, 0}}};

    mesh.devices = {dev4, dev5};

    std::unordered_set<MockChipId> relay_broken_non_mmio = {4};
    auto skip = compute_mmio_ac_skip_co(mesh, relay_broken_non_mmio);

    EXPECT_TRUE(skip.empty()) << "Non-MMIO peer should not be added to MMIO skip set.";
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIX CP tests — FIX CA failure → all MMIO ETH peers skipped
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FixCPMockTest, CaFailedDevice_AllMmioPathsSkipped) {
    auto mesh = make_t3k_topology();
    // Device 4 has 4 ETH channels (0-3), each connected to Device 0 chan 4-7.
    std::unordered_set<MockChipId> ca_failed_devices = {4};

    std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>> mmio_ac_skip_channels;
    extend_mmio_ac_skip_cp(mesh, ca_failed_devices, mmio_ac_skip_channels);

    // All 4 MMIO peers should be in the skip set.
    EXPECT_EQ(mmio_ac_skip_channels.size(), 4u);
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 4)));
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 5)));
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 6)));
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 7)));

    // Each value should record the failed non-MMIO device (4) and the channel.
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto key = make_mmio_skip_key(0, 4 + ch);
        EXPECT_EQ(mmio_ac_skip_channels[key].first, 4);
        EXPECT_EQ(mmio_ac_skip_channels[key].second, ch);
    }
}

TEST(FixCPMockTest, NoCaFailed_NoSkipsAdded) {
    auto mesh = make_t3k_topology();
    std::unordered_set<MockChipId> ca_failed_devices;  // empty

    std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>> mmio_ac_skip_channels;
    extend_mmio_ac_skip_cp(mesh, ca_failed_devices, mmio_ac_skip_channels);

    EXPECT_TRUE(mmio_ac_skip_channels.empty());
}

TEST(FixCPMockTest, PartialCaFailure_OnlyFailedDeviceSkipped) {
    auto mesh = make_t3k_topology();
    // Only device 4 failed; device 5 did not.
    std::unordered_set<MockChipId> ca_failed_devices = {4};

    std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>> mmio_ac_skip_channels;
    extend_mmio_ac_skip_cp(mesh, ca_failed_devices, mmio_ac_skip_channels);

    // Device 4's 4 MMIO peers present.
    EXPECT_EQ(mmio_ac_skip_channels.size(), 4u);

    // Device 5's MMIO peers NOT present (device 1 chans 4,5).
    EXPECT_FALSE(mmio_ac_skip_channels.count(make_mmio_skip_key(1, 4)));
    EXPECT_FALSE(mmio_ac_skip_channels.count(make_mmio_skip_key(1, 5)));
}

TEST(FixCPMockTest, CaFailedAndCOPopulated_NoDoubleCount) {
    // Both FIX CO and FIX CP apply to overlapping channels on device 4.
    // FIX CO adds skip for bc_deadlock channels; FIX CP adds skip for ALL channels.
    // emplace deduplicates — final count should be 4 (all of device 4's MMIO peers).
    auto mesh = make_t3k_topology();
    auto* dev4 = const_cast<MockDevice*>(mesh.get_device(4));
    dev4->relay_broken = true;
    dev4->bc_deadlock_channels = {0, 1};  // subset of all channels

    // Run FIX CO first.
    std::unordered_set<MockChipId> relay_broken_non_mmio = {4};
    auto mmio_ac_skip_channels = compute_mmio_ac_skip_co(mesh, relay_broken_non_mmio);

    // FIX CO should have added 2 entries (bc_deadlock chans 0, 1 -> MMIO chans 4, 5).
    ASSERT_EQ(mmio_ac_skip_channels.size(), 2u);

    // Now run FIX CP (device 4 failed FIX CA).
    std::unordered_set<MockChipId> ca_failed_devices = {4};
    extend_mmio_ac_skip_cp(mesh, ca_failed_devices, mmio_ac_skip_channels);

    // Should now have all 4 MMIO peers (emplace dedup means chans 4,5 not double-counted).
    EXPECT_EQ(mmio_ac_skip_channels.size(), 4u);
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 4)));
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 5)));
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 6)));
    EXPECT_TRUE(mmio_ac_skip_channels.count(make_mmio_skip_key(0, 7)));

    // The first two entries should retain FIX CO's non-MMIO peer info (emplace
    // does not overwrite existing keys).
    auto key4 = make_mmio_skip_key(0, 4);
    EXPECT_EQ(mmio_ac_skip_channels[key4].first, 4);   // non_mmio_id from FIX CO
    EXPECT_EQ(mmio_ac_skip_channels[key4].second, 0u);  // bc_chan from FIX CO
}

TEST(FixCPMockTest, MultipleCaFailedDevices_AllPeersSkipped) {
    auto mesh = make_t3k_topology();
    // Both device 4 and device 5 failed FIX CA.
    std::unordered_set<MockChipId> ca_failed_devices = {4, 5};

    std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>> mmio_ac_skip_channels;
    extend_mmio_ac_skip_cp(mesh, ca_failed_devices, mmio_ac_skip_channels);

    // Device 4: 4 MMIO peers (dev 0 chans 4-7)
    // Device 5: 2 MMIO peers (dev 1 chans 4-5)
    EXPECT_EQ(mmio_ac_skip_channels.size(), 6u);
}

TEST(FixCPMockTest, UnknownFailedDevice_NoEffect) {
    auto mesh = make_t3k_topology();
    // Device 99 does not exist in the mesh.
    std::unordered_set<MockChipId> ca_failed_devices = {99};

    std::unordered_map<uint64_t, std::pair<MockChipId, uint32_t>> mmio_ac_skip_channels;
    extend_mmio_ac_skip_cp(mesh, ca_failed_devices, mmio_ac_skip_channels);

    EXPECT_TRUE(mmio_ac_skip_channels.empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Key packing tests — verify make_mmio_skip_key uniqueness
// ═══════════════════════════════════════════════════════════════════════════════

TEST(MakeMMIOSkipKeyTest, DifferentChipsSameChannel_DifferentKeys) {
    EXPECT_NE(make_mmio_skip_key(0, 4), make_mmio_skip_key(1, 4));
}

TEST(MakeMMIOSkipKeyTest, SameChipDifferentChannels_DifferentKeys) {
    EXPECT_NE(make_mmio_skip_key(0, 4), make_mmio_skip_key(0, 5));
}

TEST(MakeMMIOSkipKeyTest, SameInputs_SameKey) {
    EXPECT_EQ(make_mmio_skip_key(3, 7), make_mmio_skip_key(3, 7));
}

}  // namespace tt::tt_metal::testing::test
