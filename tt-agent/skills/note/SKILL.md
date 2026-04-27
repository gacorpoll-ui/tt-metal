---
name: note
description: "Record an entry to git-tracked timeline notes at ~/.tt-agent/notes/. Used by every skill that writes notes; users invoke directly to record findings or observations. One commit per entry; subject `<topic>: <entry-title>`."
metadata:
  layer: meta
---

# TT Note

## Purpose

Writes entries to `~/.tt-agent/notes/` per a single canonical convention so
every observable agent action is auditable from `git log` alone.

## When to Invoke

- Developer asks to record a finding or observation
- Workflow skill records a per-phase / per-iter entry to its scope file
- `tt:learn`, `tt:profiler`, `tt:code-review`, `tt:orchestrator` record entries to their respective topic files

## Pipeline

```
detect topic → format entry → atomic-write → return path
```

1. **Detect topic.** Pick the filename from input per `protocol.md` § Filename rules.
2. **Format entry.** Build the H2 entry per `protocol.md` § Entry shape; capture source-repo + short-SHA per § Source-SHA capture.
3. **Atomic-write.** Prepend to the topic file (auto-init the notes repo if absent); commit per `protocol.md` § Atomic-write protocol with subject `<topic>: <entry-title>`.
4. **Return path.** Report the topic file path and notes-repo SHA.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Filename rules, file shape, entry shape, source-SHA capture, auto-init, atomic-write protocol, cross-topic referencing, operational policies | `protocol.md` |
