// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Polls dispatch/prefetch telemetry from every dispatch core on the given
// device(s) and prints a compact one-line-per-core summary every 5 seconds.
//
// Usage:
//   dispatch_telemetry_dump [--device N] [--interval-ms N] [--num-cqs N]
//
// Caveat: CreateDevice takes the chip exclusively, so this tool cannot attach
// to a chip that is already in use by another process. To observe non-zero
// counters you must enqueue work from this same tool (or convert the polling
// into a library function that user apps call). For the prototype, running
// this alone will only verify the read path end-to-end; the counters will
// stay at zero unless something is driving the command queues.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <fmt/core.h>

#include <host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/dispatch_telemetry.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

struct CoreEntry {
    std::string role;  // "PREFETCH", "PREFETCH_D", "DISPATCH", "DISPATCH_D", "DISPATCH_S"
    tt_cxy_pair cxy;
    uint8_t cq_id;
};

// Walk dispatch_core_manager and collect every allocated dispatch/prefetch core for a device.
std::vector<CoreEntry> collect_cores(IDevice* device) {
    std::vector<CoreEntry> entries;
    auto& dcm = dispatch_core_manager::instance();
    auto& cluster = MetalContext::instance().cluster();
    ChipId chip = device->id();
    uint16_t channel = cluster.get_assigned_channel_for_device(chip);
    uint8_t num_cqs = device->num_hw_cqs();

    for (uint8_t cq = 0; cq < num_cqs; ++cq) {
        if (dcm.is_prefetcher_core_allocated(chip, channel, cq)) {
            entries.push_back({"PREFETCH", dcm.prefetcher_core(chip, channel, cq), cq});
        }
        if (dcm.is_prefetcher_d_core_allocated(chip, channel, cq)) {
            entries.push_back({"PREFETCH_D", dcm.prefetcher_d_core(chip, channel, cq), cq});
        }
        if (dcm.is_dispatcher_core_allocated(chip, channel, cq)) {
            entries.push_back({"DISPATCH", dcm.dispatcher_core(chip, channel, cq), cq});
        }
        if (dcm.is_dispatcher_d_core_allocated(chip, channel, cq)) {
            entries.push_back({"DISPATCH_D", dcm.dispatcher_d_core(chip, channel, cq), cq});
        }
        if (dcm.is_dispatcher_s_core_allocated(chip, channel, cq)) {
            entries.push_back({"DISPATCH_S", dcm.dispatcher_s_core(chip, channel, cq), cq});
        }
    }
    return entries;
}

bool is_prefetch_role(const std::string& role) { return role.rfind("PREFETCH", 0) == 0; }

void print_snapshot(IDevice* device, const std::vector<CoreEntry>& entries) {
    auto core_type = dispatch_core_manager::instance().get_dispatch_core_type();

    fmt::print("\033[2J\033[H");  // clear + home
    fmt::print(
        "dispatch_telemetry_dump  chip={}  num_hw_cqs={}  cores={}  ts={}s\n",
        device->id(),
        device->num_hw_cqs(),
        entries.size(),
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    fmt::print("{:<11} {:>3} {:>10} {:>10} {:>12} {:>10}\n", "role", "cq", "core", "blocked", "unblocked", "cmds");

    for (const auto& e : entries) {
        CoreCoord logical{e.cxy.x, e.cxy.y};
        std::string core_str = fmt::format("({},{})", logical.x, logical.y);
        if (is_prefetch_role(e.role)) {
            auto t = read_prefetch_telemetry(device, logical, core_type);
            if (t) {
                fmt::print(
                    "{:<11} {:>3} {:>10} {:>10} {:>12} {:>10}\n",
                    e.role,
                    e.cq_id,
                    core_str,
                    t->blocked_by_host_count,
                    t->unblocked_by_host_count,
                    t->command_count);
            } else {
                fmt::print("{:<11} {:>3} {:>10} <invalid>\n", e.role, e.cq_id, core_str);
            }
        } else {
            auto t = read_dispatch_telemetry(device, logical, core_type);
            if (t) {
                fmt::print(
                    "{:<11} {:>3} {:>10} {:>10} {:>12} {:>10}\n",
                    e.role,
                    e.cq_id,
                    core_str,
                    t->blocked_by_host_count,
                    t->unblocked_by_host_count,
                    "-");
            } else {
                fmt::print("{:<11} {:>3} {:>10} <invalid>\n", e.role, e.cq_id, core_str);
            }
        }
    }
    std::cout.flush();
}

}  // namespace

int main(int argc, char** argv) {
    ChipId device_id = 0;
    uint8_t num_hw_cqs = 1;
    int interval_ms = 5000;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--device" || a == "-d") && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (a == "--interval-ms" && i + 1 < argc) {
            interval_ms = std::atoi(argv[++i]);
        } else if (a == "--num-cqs" && i + 1 < argc) {
            num_hw_cqs = static_cast<uint8_t>(std::atoi(argv[++i]));
        } else if (a == "--help" || a == "-h") {
            fmt::print("Usage: {} [--device N] [--interval-ms N] [--num-cqs N]\n", argv[0]);
            return 0;
        }
    }

    IDevice* device = CreateDevice(device_id, num_hw_cqs);
    auto entries = collect_cores(device);
    if (entries.empty()) {
        fmt::print(stderr, "No dispatch/prefetch cores found on chip {}\n", device_id);
        CloseDevice(device);
        return 1;
    }

    fmt::print("Polling {} cores every {}ms. Ctrl-C to quit.\n", entries.size(), interval_ms);
    while (true) {
        print_snapshot(device, entries);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    CloseDevice(device);
    return 0;
}
