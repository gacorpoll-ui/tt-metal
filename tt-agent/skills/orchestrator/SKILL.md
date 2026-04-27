---
name: orchestrator
description: "Route, plan, and decompose high-level Tenstorrent hardware development requests — new kernels, ops, models, CI failures, regressions, optimizations — into sequenced skill dispatches with tracked plans"
metadata:
  layer: orchestration
---

# TT Orchestrator

## Purpose

Translates a developer's high-level intent into a structured plan, then dispatches to
the right tt-agent skills in the right order. The orchestrator never does the work
itself — it scopes, decomposes, sequences, and verifies.

## When to Invoke

Trigger on high-level requests like:
- "Implement a new op for X"
- "Optimize the performance of Y"
- "This CI job is failing, fix it"
- "Debug why model Z is producing wrong results"
- "Profile the matmul kernel and improve throughput"
- "Add a new model to tt-metal"

If the request maps directly to a single tool (e.g., "just run this pytest"), go there
directly. Use the orchestrator when the request requires multiple steps or skills.

## Pipeline

```
analyze → scope → decompose → dispatch → verify → iterate
```

1. **Analyze**: Read the request. Classify as: new build, optimize existing, fix failure,
   or investigate regression. Load `decomposer.md` for decomposition patterns.

2. **Scope**: Identify the target — kernel, op, model, or pipeline. Determine which
   hardware tier is involved (bare-metal kernel, ttnn op, model layer).

3. **Decompose**: Break the request into ordered sub-tasks. Prepend a `Plan — v1`
   entry to `~/.tt-agent/notes/<task-slug>.md`. See `decomposer.md` for standard
   patterns.

4. **Dispatch**: Execute sub-tasks by invoking the appropriate skills (see table below).
   Prepend `Dispatch —` and `Status —` entries to `<task-slug>.md` as work
   progresses.

5. **Verify**: After each major step, confirm outputs meet the TT quality bar:
   PCC > 0.999 vs PyTorch reference, CB sizing fits L1, tile alignment correct.

6. **Iterate**: If verification fails, re-enter at step 4 with updated context.
   If stuck after 3 iterations, escalate with a detailed status report.

## Skill Dispatch Table

| Situation | Dispatch to |
|---|---|
| Build, run, or execute on device | `/tt:run` |
| Profile device time on a kernel or op | `/tt:profiler` |
| Optimize throughput toward a goal (iterative) | `/tt:optimizer` |
| Regression recovery (was working, now broken) | `/tt:recover` |
| Kernel-level diagnosis (hang, crash, RISC-V state) | `/tt:debugger` |
| Write or run tests for an op/kernel/model | `/tt:tester` |
| Need codebase context before proceeding | `/tt:learn` |
| Record a finding or observation | `/tt:note` |
| Design a new op or kernel | `/tt:designer` (future) |
| Review code before merging | `/tt:code-review` |

## Document Protocol

Each high-level task gets one timeline file at `~/.tt-agent/notes/<task-slug>.md`.
All plan, dispatch, and status updates are entries in this single file — no
separate plan/status files. Entries are written via `/tt:note`.

Conventional entry kinds for orchestrator:

| Entry title prefix | Purpose |
|---|---|
| `Plan — <revision>` | Ordered sub-tasks + decision rationale (latest entry carries the active plan) |
| `Dispatch — <skill> on <scope>` | Pre-dispatch context: target, expected output |
| `Status — <step>` | Progress, blockers, findings so far (latest entry carries current status) |
| `Done — <summary>` | Final outcome of the task |

The latest entry (top of file) carries the active plan and current status — a
single read shows where the task stands. Older entries preserve the history of
how it got there.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Decomposing new vs optimize vs fix | `decomposer.md` |
