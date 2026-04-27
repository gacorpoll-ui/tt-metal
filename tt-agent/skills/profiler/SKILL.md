---
name: profiler
description: "Profile a TT kernel or op with Tracy + tt-perf-report — runs through tt:run (MCP), locates the CSV, interprets device timing, produces a structured profile note. Use for device-time breakdowns and bottleneck identification."
metadata:
  layer: tool
---

# TT Profiler

## Purpose

Own the automation and interpretation of a single profiling cycle.
Invocation mechanics (tracy command, env conflicts, output paths, pandas
fallback) live in `knowledge/recipes/tt-metal/profiler.md` — this skill
consumes that recipe, it doesn't restate it.

Pure measurement — never modifies code, never iterates, never hypothesizes
fixes. Called by `tt:optimizer` once per iteration; invokable directly by
a developer for a one-shot profile.

## When to Invoke

- "Profile this test / kernel / op"
- "What's the device time for X?"
- "Where's the bottleneck in <test>?"
- Called by `tt:optimizer` each iteration (baseline and trials)

## Pipeline

```
resolve target → compose + dispatch → find CSV → analyze → interpret → note
```

1. **Resolve target**: caller hands over a pytest path (+ optional `-k`
   filter) or a binary target. Confirm the file exists.
2. **Compose + dispatch**: build the tracy invocation per the recipe and
   hand it to `tt:run`. tt:run routes to tt-device-mcp — profiling runs
   on-device and must not go through Bash. Surface any caller env vars
   the recipe lists as conflicting before running.
3. **Find CSV**: locate the most recent subdir under the reports path
   (recipe § Output layout) by mtime. If missing, stop and report;
   recipe § Constraints lists the common causes.
4. **Analyze**: run `tt-perf-report`. On crash, use the pandas fallback
   the recipe documents.
5. **Interpret**: load `interpretation.md` for columns, bound classes,
   and tags.
6. **Note**: invoke `/tt:note` with topic=`<scope>` (caller's scope, or
   developer-supplied if standalone), title=`Profile — <summary with dominant
   metric and headline utilization>`, body=<top ops, bottleneck, CSV path>.

## Scope Naming

`<scope>` is a short slug — op, test, or kernel name. Lowercase, dash-separated.

## Entry body convention

```markdown
**Top ops** (device-FW μs):
| Op | Time | Cores | DRAM% | FLOPs% | Bound |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

**Bottleneck:** <which op + why>
**CSV:** `<path>` (full Tracy output)
```

## Caller Contract

- **Input**: target (pytest path + filter, or binary) and scope slug.
- **Output**: path to the profile note, plus top-3 op codes and their
  device durations inline for the caller's convenience.
- **Failure**: if profiling data can't be collected, return an error
  naming the likely cause. Report only observed numbers.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Tracy command, env constraints, output paths, pandas fallback | `knowledge/recipes/tt-metal/profiler.md` |
| Reading CSV columns, bottleneck classification | `interpretation.md` |
| MCP routing, env handoff | invoke `tt:run` |
| Unfamiliar op type or perf idiom | invoke `tt:learn("<op> profiling patterns")` |
