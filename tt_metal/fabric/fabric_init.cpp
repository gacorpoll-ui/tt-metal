// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>
#include <unordered_set>

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_fabric {

// Thread-local configure-cores inject seam — default-constructed (empty std::function) on every
// thread.  Only set by tests via set_configure_cores_inject_fn() before MeshDevice::create().
thread_local ConfigureFabricCoresInjectFn s_configure_cores_inject_fn_;

void set_configure_cores_inject_fn(ConfigureFabricCoresInjectFn fn) {
    s_configure_cores_inject_fn_ = std::move(fn);
}

void clear_configure_cores_inject_fn() {
    s_configure_cores_inject_fn_ = {};
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program(tt::tt_metal::IDevice* device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Guard: if this device's physical chip is not in the fabric cluster mapping, skip
    // fabric program compilation rather than crashing with TT_FATAL inside FabricBuilder.
    // This can happen when hardware connectivity issues cause auto-discovery to downgrade
    // to a smaller mesh (e.g. 2x2 instead of 2x4) that excludes some physical chips.
    // The device will simply not participate in fabric routing for this session.
    if (!control_plane.is_physical_chip_in_fabric_cluster(device->id())) {
        log_warning(
            tt::LogFabric,
            "create_and_compile_tt_fabric_program: Physical chip {} is not in the fabric cluster "
            "chip mapping; skipping fabric program compilation. This typically indicates hardware "
            "connectivity issues that caused auto-discovery to select a smaller mesh topology. "
            "The device will not participate in fabric routing.",
            device->id());
        return nullptr;
    }

    auto fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();

    auto& fabric_context = control_plane.get_fabric_context();

    // Use FabricBuilder to coordinate the build phases
    FabricBuilder builder(device, *fabric_program_ptr, fabric_context);

    // Execute build phases
    builder.discover_channels();
    builder.create_routers();
    if (!builder.has_routers()) {
        return nullptr;
    }

    builder.connect_routers();
    builder.compile_ancillary_kernels();
    builder.create_kernels();

    // Compile the program
    tt::tt_metal::detail::CompileProgram(
        device, *fabric_program_ptr, tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch());

    return fabric_program_ptr;
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device) {
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        return create_and_compile_tt_fabric_program(device);
    }
    return nullptr;
}

FabricCoresHealth configure_fabric_cores(
    tt::tt_metal::IDevice* device,
    const std::unordered_set<uint32_t>& pre_known_dead_channels,
    const std::unordered_set<uint32_t>& skip_soft_reset_channels) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto soc_desc = cluster.get_soc_desc(device->id());
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto addresses_to_clear = builder_context.get_fabric_router_addresses_to_clear();
    const auto& router_config = builder_context.get_fabric_router_config();
    std::vector<uint32_t> router_zero_buf(router_config.router_buffer_clear_size_words, 0);

    // Diagnostic: log entry state so CI logs show exactly what configure_fabric_cores sees.
    log_info(
        tt::LogMetal,
        "configure_fabric_cores: Device {} ({}) — {} active channels, {} pre_known_dead, "
        "{} skip_soft_reset (base-UMD relay). [AUDIT #42429]",
        device->id(),
        device->is_mmio_capable() ? "MMIO" : "non-MMIO",
        router_chans_and_direction.size(),
        pre_known_dead_channels.size(),
        skip_soft_reset_channels.size());

    // Fix #42429: After a cancelled CI run, SIGKILL, or normal AllGather CCL teardown, the
    // ERISC BRISC may be halted.  When an ERISC fabric router self-terminates (writes
    // EDMStatus::TERMINATED to its sync address and halts BRISC), subsequent L1 firmware
    // writes in this function have no effect because BRISC never executes the new code.
    // This causes the Phase 5 health check in quiesce_and_restart_fabric_workers() to fail
    // with TERMINATED ≠ READY_FOR_TRAFFIC on the second (post-AllGather) quiesce cycle.
    //
    // Fix: perform a BRISC-only soft reset (assert + deassert) before writing L1 to ensure
    // BRISC is live when the new fabric firmware is loaded.
    //
    // Safety: resetting only ERISC0/BRISC keeps the subordinate ERISC (NCRISC) running,
    // which maintains the ETH PHY link.  The reset window is brief (PCIe round-trip,
    // microseconds) and mirrors the pattern used in risc_firmware_initializer.cpp
    // reset_cores().  Although base UMD firmware is not expected to touch fabric-specific
    // state addresses, the L1 clear below (addresses_to_clear) zeroes edm_status_address,
    // termination_signal_address, edm_local_sync_address, and edm_local_tensix_sync_address
    // as a belt-and-suspenders measure — so even if base firmware or stale L1 leaves garbage,
    // the next session's terminate_stale_erisc_routers() sees clean zeros.
    // Track dead channels so the L1 clear loop below can skip them.
    // For non-MMIO chips, WriteToDeviceL1 routes writes through ethernet — the same path
    // that timed out during the soft reset — and can hang indefinitely (write_core has no
    // per-call timeout unlike read_non_mmio).  Skipping the L1 clear for dead channels is
    // safe: firmware won't start on them regardless of whether L1 was zeroed.
    //
    // Seed with pre_known_dead_channels: channels where the probe L1 read in
    // terminate_stale_erisc_routers() threw an exception (physically dead link — ERISC
    // completely unresponsive).  For these channels we skip assert_risc_reset_at_core()
    // entirely because the ETH relay path is non-functional.
    //
    // Note: channels with corrupt/garbage L1 status (probe read succeeded but value was
    // not a valid EDMStatus) are NOT in this set — their probe read succeeded, proving
    // the relay path works, so assert_risc_reset_at_core() should also succeed.  Those
    // channels proceed through normal soft reset here.
    std::unordered_set<uint32_t> dead_channels = pre_known_dead_channels;
    // Track channels that NEWLY failed in this call (not pre-known).
    // This lets the caller (configure_fabric) distinguish "expected degraded mode" from
    // "unexpected new failure" — the former warrants a warning, the latter a hard throw.
    std::unordered_set<uint32_t> newly_dead_channels;
    // FIX RR (#42429): Track pre_known channels that were successfully recovered by
    // PCIe-direct soft reset.  Returned to configure_fabric() so it can subtract them from
    // pre_dead_channels and load firmware on those channels instead of skipping them.
    std::unordered_set<uint32_t> recovered_channels;
    // Note: all_channels_healthy is computed at the END of this function (after FIX RR may
    // have recovered pre_known_dead channels from dead_channels).  Do not set it here.
    if (!pre_known_dead_channels.empty()) {
        if (device->is_mmio_capable()) {
            // FIX RR (#42429): MMIO devices have PCIe-direct access — the relay-queue-fill
            // concern that motivates skipping assert_risc_reset_at_core for non-MMIO channels
            // does NOT apply here.  ROM-postcode channels (0x49705180) that timed out in FIX RP
            // are still accessible via PCIe; a soft reset kicks them out of ROM phase so they
            // boot to base firmware instead of persisting as permanently degraded every session.
            // We attempt the reset and catch failures — if PCIe truly is broken for a channel
            // we mark it newly_dead; otherwise we clear it from dead_channels so the L1 write
            // loop below can zero its sync addresses and normal firmware launch can proceed.
            log_info(
                tt::LogMetal,
                "configure_fabric_cores: Device {} (MMIO) — attempting PCIe-direct soft reset "
                "for {} pre-confirmed problematic channels (FIX RR #42429). "
                "ROM-postcode channels that timed out in FIX RP can be recovered this way.",
                device->id(),
                pre_known_dead_channels.size());
        } else {
            log_warning(
                tt::LogMetal,
                "configure_fabric_cores: Device {} skipping assert_risc_reset_at_core for {} "
                "pre-confirmed problematic channels (probe timed out or L1 corrupt in "
                "terminate_stale_erisc_routers). This avoids the indefinite hang in read_non_mmio "
                "when the 4-slot relay ETH queue fills after dead channels exhaust UMD timeouts.",
                device->id(),
                pre_known_dead_channels.size());
        }
    }
    // FIX RR (#42429): MMIO device — attempt PCIe-direct soft reset for all dead channels.
    // FIX DC (#42429): Two-pass: assert+deassert all first, then parallel-poll.
    // Sequential cost: N × 500ms. Parallel cost: 500ms regardless of N.
    struct FIX_BH_PollState {
        uint32_t router_chan;
        tt_cxy_pair core_loc;
        bool booted = false;
        bool done = false;
    };
    std::vector<FIX_BH_PollState> bh_poll_states;

    {
        const auto chip_id = device->id();
        for (const auto& [router_chan, _] : router_chans_and_direction) {
            // Skip channels already confirmed dead by terminate_stale_erisc_routers().
            // For non-MMIO devices, calling assert_risc_reset_at_core() on dead channels can
            // hang indefinitely because the UMD relay queue fills after the channel exhausts
            // its retry budget.  For MMIO devices, PCIe-direct access is available so we
            // attempt the reset below (FIX RR) and only skip on genuine PCIe failure.
            if (dead_channels.count(router_chan)) {
                if (!device->is_mmio_capable()) {
                    log_debug(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} is pre-confirmed dead "
                        "(non-MMIO) — skipping soft reset to avoid relay queue fill",
                        chip_id,
                        router_chan);
                    continue;
                }
                // FIX RR (#42429): MMIO device — attempt PCIe-direct soft reset.
                // FIX DC (#42429): Pass 1 — assert + write fw_launch_addr + deassert only.
                // Polling moved to Pass 2 (parallel) below to avoid N × 500ms sequential wait.
                try {
                    auto virtual_core = cluster.get_virtual_eth_core_from_channel(chip_id, router_chan);
                    tt_cxy_pair core_loc(chip_id, virtual_core);
                    cluster.assert_risc_reset_at_core(core_loc, tt::umd::RiscType::ERISC0);
                    // FIX BO (#42429): restore fw_launch_addr to base-UMD firmware entry point
                    // BEFORE deassert.  FIX BN clears fw_launch_addr=0 during teardown heartbeat
                    // confirmation.  If FIX RR then resets such a channel, ERISC ROM reads
                    // fw_launch_addr==0 and loops at 0x49705180 indefinitely — FIX BH then times
                    // out (500ms) and the channel stays dead.  Writing the valid entry point
                    // while ERISC is held in reset guarantees ROM can exit immediately on deassert.
                    try {
                        const auto& hal_bo = tt::tt_metal::MetalContext::instance().hal();
                        const auto aeth_idx_bo = hal_bo.get_programmable_core_type_index(
                            tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
                        const auto& jit_cfg_bo = hal_bo.get_jit_build_config(aeth_idx_bo, 0, 0);
                        if (jit_cfg_bo.fw_launch_addr_value != 0) {
                            cluster.write_core_immediate(
                                chip_id,
                                virtual_core,
                                std::vector<uint32_t>{jit_cfg_bo.fw_launch_addr_value},
                                jit_cfg_bo.fw_launch_addr);
                        }
                    } catch (...) {
                        // Non-fatal — deassert proceeds; FIX BH may still see success if
                        // fw_launch_addr was not previously zeroed.
                    }
                    cluster.deassert_risc_reset_at_core(core_loc, tt::umd::RiscType::ERISC0);
                    // FIX DC (#42429): Collect this channel for parallel boot poll (Pass 2).
                    // Polling is deferred so all channels boot concurrently under a shared deadline.
                    bh_poll_states.push_back({router_chan, core_loc, false, false});
                } catch (const std::exception& e) {
                    // FIX BH-DEGRADE: Channel was pre-confirmed dead; soft reset failure keeps it
                    // dead (degraded mode), not newly_dead (TT_THROW).
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} FIX RR — PCIe-direct "
                        "soft reset FAILED ({}). Channel remains in dead_channels (degraded mode).",
                        chip_id,
                        router_chan,
                        e.what());
                } catch (...) {
                    // FIX BH-DEGRADE: same as above.
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} FIX RR — PCIe-direct "
                        "soft reset FAILED (unknown exception). Channel remains in dead_channels (degraded mode).",
                        chip_id,
                        router_chan);
                }
                continue;
            }

            // FIX M (#42429): Skip soft reset for channels with base-UMD relay firmware.
            // Their BRISC is running and serves as the ETH relay endpoint for non-MMIO reads.
            // assert_risc_reset halts the BRISC → deassert_risc_reset (relay read) times out
            // → all subsequent reads from MMIO→non-MMIO fail → cascade hang.
            // write_launch_msg_to_core transitions this firmware to fabric firmware without a reset.
            // NOTE: the L1 clear loop below is ALSO skipped for these channels — see FIX TG.
            if (skip_soft_reset_channels.count(router_chan)) {
                log_info(
                    tt::LogMetal,
                    "configure_fabric_cores: device {} channel {} base-UMD relay firmware "
                    "(0x49706550) — skipping soft reset (launch_msg handles transition, "
                    "halting BRISC would kill the ETH relay) [FIX M #42429]",
                    chip_id,
                    router_chan);
                continue;
            }

            try {
                // Test seam (Scenario W): if set, call the inject function instead of (or
                // before) the real assert_risc_reset_at_core().  If it throws, the catch
                // block below is exercised identically to a real UMD failure.
                if (s_configure_cores_inject_fn_) {
                    s_configure_cores_inject_fn_(device, router_chan);
                    // If the inject fn returned without throwing, skip the real hardware
                    // call — the seam fully replaces assert/deassert for this channel.
                    continue;
                }

                // get_virtual_eth_core_from_channel returns the virtual CoreCoord needed
                // for tt_cxy_pair, which is what assert/deassert_risc_reset_at_core expect.
                auto virtual_core = cluster.get_virtual_eth_core_from_channel(chip_id, router_chan);
                tt_cxy_pair core_loc(chip_id, virtual_core);

                // Assert ERISC0 (== BRISC) reset — halts only the main ERISC processor.
                // The subordinate ERISC continues running and maintains the ETH PHY link.
                cluster.assert_risc_reset_at_core(core_loc, tt::umd::RiscType::ERISC0);

                // Immediately deassert so ERISC0 restarts into base UMD firmware.
                // The window where ERISC0 is halted is limited to the PCIe write round-trip.
                cluster.deassert_risc_reset_at_core(core_loc, tt::umd::RiscType::ERISC0);

                log_debug(
                    tt::LogMetal,
                    "configure_fabric_cores: ERISC0 soft reset bounce on device {} channel {} "
                    "(BRISC halt recovery)",
                    chip_id,
                    router_chan);
            } catch (const std::exception& e) {
                // Fatal for this channel: remote chip is unreachable.
                // Skip L1 writes for this channel — WriteToDeviceL1 on a non-MMIO chip routes
                // through ethernet and will hang indefinitely on a dead channel.
                dead_channels.insert(router_chan);
                newly_dead_channels.insert(router_chan);
                log_warning(
                    tt::LogMetal,
                    "configure_fabric_cores: Failed ERISC0 soft reset on device {} channel {}: {}. "
                    "Skipping L1 clear for this channel — firmware will not start on it.",
                    chip_id,
                    router_chan,
                    e.what());
            } catch (...) {
                dead_channels.insert(router_chan);
                newly_dead_channels.insert(router_chan);
                log_warning(
                    tt::LogMetal,
                    "configure_fabric_cores: Failed ERISC0 soft reset on device {} channel {} "
                    "(unknown exception). Skipping L1 clear for this channel.",
                    chip_id,
                    router_chan);
            }
        }
    }

    // FIX DC (#42429) FIX BH Pass 2: parallel-poll all MMIO dead channels that were deasserted
    // in Pass 1.  All channels boot concurrently under a single shared deadline.
    // Sequential cost: N × 500ms (e.g. 24 × 500ms = 12s). Parallel cost: 500ms regardless of N.
    if (!bh_poll_states.empty()) {
        constexpr uint32_t kRomPostcode_BH = 0x49705180u;
        constexpr uint32_t kFIX_BH_BootWaitMs = 500;
        constexpr uint32_t kFIX_BH_PollIntervalMs = 5;
        const uint64_t edm_addr_bh = static_cast<uint64_t>(router_config.edm_status_address);
        const auto chip_id_bh = device->id();
        log_info(
            tt::LogMetal,
            "FIX DC (#42429) FIX BH: parallel-polling {} MMIO dead channel(s) for ROM exit "
            "(shared {}ms deadline, dev={}).",
            bh_poll_states.size(),
            kFIX_BH_BootWaitMs,
            chip_id_bh);
        const auto bh_deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(kFIX_BH_BootWaitMs);
        while (std::chrono::steady_clock::now() < bh_deadline) {
            bool all_done = true;
            for (auto& st : bh_poll_states) {
                if (st.done) continue;
                all_done = false;
                std::vector<uint32_t> status_buf(1, 0);
                try {
                    cluster.read_core(status_buf, sizeof(uint32_t), st.core_loc, edm_addr_bh);
                } catch (...) {
                    // FIX DD (#42429): Transient PCIe read error during parallel poll.
                    // Keep polling — channel may recover on next iteration.  If all iterations
                    // fail, the channel stays in dead_channels (degraded mode).
                    continue;
                }
                if (status_buf[0] != kRomPostcode_BH) {
                    st.booted = true;
                    st.done = true;
                    log_info(
                        tt::LogMetal,
                        "FIX DC (#42429) FIX BH: device {} channel {} booted from ROM to 0x{:08x}.",
                        chip_id_bh,
                        st.router_chan,
                        status_buf[0]);
                }
            }
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(kFIX_BH_PollIntervalMs));
        }
        for (auto& st : bh_poll_states) {
            if (st.booted) {
                // Boot succeeded — clear from dead set so L1 init proceeds and
                // configure_fabric() loads firmware on this channel.
                dead_channels.erase(st.router_chan);
                recovered_channels.insert(st.router_chan);
                log_info(
                    tt::LogMetal,
                    "FIX DC (#42429) FIX RR+BH: device {} channel {} recovered (parallel poll). "
                    "Cleared from dead_channels; L1 init will proceed.",
                    chip_id_bh,
                    st.router_chan);
            } else {
                // FIX BH-DEGRADE (#42429): ROM boot did not complete within 500ms.
                // This channel was pre-confirmed dead by FIX BT — staying dead after a failed
                // recovery attempt is "still dead", not a new failure.  Leave dead_channels intact.
                log_warning(
                    tt::LogMetal,
                    "FIX DC (#42429) FIX BH: device {} channel {} did not exit ROM within {}ms. "
                    "Remains in dead_channels (degraded mode).",
                    chip_id_bh,
                    st.router_chan,
                    kFIX_BH_BootWaitMs);
            }
        }
    }

    for (const auto& [router_chan, _] : router_chans_and_direction) {
        if (dead_channels.count(router_chan)) {
            // Skip L1 clear for dead channels — WriteToDeviceL1 on non-MMIO chips routes
            // writes through ethernet and will hang if the channel is unreachable.
            //
            // Cascade prevention (#42429): for corrupt-but-reachable channels (probe read
            // succeeded in terminate_stale_erisc_routers, but status was garbage),
            // terminate_stale_erisc_routers now zeroes edm_status_address before adding
            // the channel to probe_dead_channels.  This means the NEXT session sees
            // edm_status=0 ("clean") and does not add the channel to pre_known_dead_channels,
            // so this L1 clear loop will process it normally — breaking the infinite cascade
            // where corrupt status persisted across container restarts on bare metal.
            continue;
        }
        // FIX TG (#42429): For base-UMD relay channels, skip ALL L1 clears and preserve the
        // edm_status_address sentinel (0x49706550) so the next session's
        // terminate_stale_erisc_routers() can identify base-UMD state and fire FIX M
        // (launch_msg transition).
        //
        // FIX TG2 was reverted (#42429): FIX TG2 attempted partial L1 clears (zeroing
        // edm_local_sync_address, edm_local_tensix_sync_address, termination_signal_address
        // while preserving edm_status_address) to fix stale ring-sync state after tt-smi -r.
        // However, these writes overwrite L1 locations that the base-UMD relay firmware uses
        // for its own protocol (the relay ERISC's live state is at those offsets).  On MMIO
        // devices, device 0 chan=8 is the UMD relay; FIX TG2 PCIe-direct writes to its L1
        // at termination_signal_address / edm_local_sync_address corrupted the relay protocol
        // state, causing wait_for_non_mmio_flush to time out with 5002ms elapsed on the next
        // non-MMIO write (observed in CI run 25949429100).
        //
        // The stale ring-sync fix must be handled inside the fabric ERISC firmware itself
        // (zeroing its own sync variables on startup) — not here where we cannot safely
        // distinguish our fabric layout from the base-UMD relay firmware's layout.
        if (skip_soft_reset_channels.count(router_chan)) {
            continue;
        }
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }

    // FIX RR (#42429): re-evaluate all_channels_healthy after FIX RR may have recovered
    // pre_known_dead channels.  If all pre-known dead channels were recovered (dead_channels
    // is now empty) and no new failures occurred, all channels are healthy.
    const bool all_channels_healthy = dead_channels.empty() && newly_dead_channels.empty();

    // Diagnostic: log exit state so CI logs show the outcome of configure_fabric_cores.
    log_info(
        tt::LogMetal,
        "configure_fabric_cores: Device {} — exit: healthy={}, dead={}, newly_dead={}, recovered={}. [AUDIT #42429]",
        device->id(),
        all_channels_healthy,
        dead_channels.size(),
        newly_dead_channels.size(),
        recovered_channels.size());

    return FabricCoresHealth{all_channels_healthy, std::move(newly_dead_channels), std::move(recovered_channels)};
}

}  // namespace tt::tt_fabric
