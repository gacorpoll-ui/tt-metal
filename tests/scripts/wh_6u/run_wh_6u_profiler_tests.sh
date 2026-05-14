#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

run_profiling_test() {
    # TEMP (revert before review): skipping the two existing tests while
    # iterating on test_realtime_profiler_perf_llama_tg to shorten CI cycle.
    # TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360
    #
    # Cross-reference real-time profiler durations against device profiler
    # on a full TG (8x4) mesh. This test was consolidated into the unified
    # real-time profiler test suite (tests/ttnn/tracy/test_realtime_profiler.py)
    # under the name test_cross_reference_tg; the old standalone file
    # test_profiler_cross_reference_TG.py no longer exists.
    # TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_realtime_profiler.py::test_cross_reference_tg --timeout 2400

    # Dispatch-zone latency gate: p50/p99 budgets for push_entry_to_host
    # and signal_realtime_profiler_and_switch under a Llama-70B workload.
    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_realtime_profiler.py::test_realtime_profiler_perf_llama_tg --timeout 3600
}

main() {
    cd $TT_METAL_HOME

    TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'

    if [[ -z "$ARCH_NAME" ]]; then
        echo "Must provide ARCH_NAME in environment" 1>&2
        exit 1
    fi

    echo "Make sure this test runs in a build with cmake option ENABLE_TRACY=ON"

    if [[ -z "$DONT_USE_VIRTUAL_ENVIRONMENT" ]]; then
        source python_env/bin/activate
    fi

    run_profiling_test
}

main "$@"
