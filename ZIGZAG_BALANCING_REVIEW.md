# Causal Zigzag Balancing Readiness

This is the current follow-up list from the causal ring-joint SDPA zigzag review.

## Resolved

- Fixed the causal ring-joint runtime hang.
  - Even per-head Q chunk counts now use pair-aligned zigzag Q distribution, preserving the fast skipped-Q path without desynchronizing K/V multicast chains.
  - Odd per-head Q chunk counts use the reader's chain-bypass fallback for skipped causal ring iterations so active cores fetch K/V directly instead of waiting on a bypassed chain hop.
  - The causal smoke test now passes:
    `scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy[mla_100k-q160-k256] -s`

- Clarified causal + joint attention.
  - The op validation rejects causal + joint attention today.
  - Added a regression test for this contract:
    `scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_rejects_causal_joint_attention -s`

- Collapsed duplicate ring-joint zigzag compile-time flags.
  - Reader, writer, and compute now consume a single `use_zigzag_balancing` flag for both Q-index remapping and causal skip behavior.
  - The host factory no longer passes two identical zigzag flags into the ring-joint kernels.

- Made the causal Q skip boundary explicit.
  - Reader, writer, and ring-joint compute derive the skip boundary from `num_local_q_chunks / 2`.
  - Shared compute helpers still default to `num_q_chunks / 2`, but ring-joint passes the local causal boundary explicitly.

- Restored the default Blackhole worker L1 setup.
  - Removed the temporary 4 KiB extra kernel-code reserve from ring-joint SDPA and DeepSeek MLA tests.
  - Causal even and odd ring-joint smoke coverage passed with a fresh `TT_METAL_CACHE` and `0/64` JIT cache hits, confirming the kernels fit without the worker-L1 reduction.

## Blockers

- Align the causal ring-joint layout contract.
  - The op now enables zigzag whenever `args.is_causal` is true in `ring_joint_sdpa_program_factory.cpp`.
  - Higher-level DeepSeek paths still expose `is_balanced=False` and can prepare sequential input/RoPE/cache/output mappings.
  - Either force causal ring-joint callers to prepare zigzag layout, or remove/rename the remaining layout flag so sequential causal layout cannot accidentally call the always-zigzag kernel.

## Correctness Cleanup

- Clean up remaining naming in tests/model code.
  - `is_balanced` now means physical zigzag layout in Python-side data prep, not an op mode.
  - Rename to something like `use_zigzag_layout` where practical to avoid confusing it with the removed op argument.

## Validation Plan

- Re-run build:
  - `./build_metal.sh --release`
  - Current status: passed after the default worker-L1 code-size cleanup.

- Re-run the causal ring-joint smoke test first:
  - `scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy[mla_100k-q160-k256] -s`
  - Current status: passed.

- Re-run remaining causal ring-joint MLA chunks and odd chunk tests:
  - `mla_100k-q160-k160`, `mla_100k-q160-k320`
  - `mla_100k-odd-total-qchunks-q128-k256`, `mla_100k-odd-total-qchunks-q128-k320`
  - Current status: passed.

- Re-run post-cleanup targeted coverage:
  - `scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy[mla_100k-q160-k320]' 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_odd_num_q_chunks[mla_100k-odd-total-qchunks-q128-k320]' -s`
  - `env TT_METAL_CACHE=/tmp/tt_metal_cache_ring_joint_codesize_<fresh> scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy[mla_100k-q160-k320]' 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_odd_num_q_chunks[mla_100k-odd-total-qchunks-q128-k320]' -s`
  - `scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_rejects_causal_joint_attention -s`
  - `scripts/run_safe_pytest.sh 'models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa[blackhole-zigzag-rpxup-2x2-line-1link-skip_pcc-no_trace-single_run-1-128-1-576-128-seq100k-q_bf16_kv_bf8]' -s`
  - Current status: passed. The fresh-cache code-size run passed with `0/64` JIT cache hits.

- Then run the requested SDPA suites:
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa`
    - Current status: passed, 29 passed and 3 skipped.
  - `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/sdpa/`
    - Current status: not completed in this pass; the directory collected 1813 cases and was stopped after initial passing MLA decode/prefill coverage because it is too broad for this ring-joint iteration.
