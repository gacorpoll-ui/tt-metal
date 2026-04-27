---
name: learn
description: "Research live Tenstorrent codebases on demand — produces dated context notes from local code search and deepwiki, consumed by other skills and developers"
metadata:
  layer: meta
---

# TT Learn

## Purpose

Researches live Tenstorrent codebases and produces dated context notes. Other skills
invoke tt-learn when they need volatile knowledge (APIs, patterns, architecture, test
coverage) before proceeding. Developers invoke it directly to build understanding
before starting work. Works in any TT repo — detects context from the working directory.

tt-learn never guesses. It reads code, synthesizes findings, and writes them down.

## When to Invoke

- Another skill needs codebase context before it can proceed
- Developer wants to understand a subsystem before modifying it
- `tt-learn("ttnn op structure")`, `tt-learn("CCL teardown sequence")`
- "I need to understand how X works"
- Orchestrator dispatches here for the "need codebase context" row

## Pipeline

```
check existing → load references → research subagent → write entry → return entry
```

1. **Check existing**: Look for an existing entry in `~/.tt-agent/notes/learn-<subject-slug>.md`.
   If the file exists and no refresh requested, return the latest entry's body
   immediately.

2. **Load references**: Scan `tt-agent/knowledge/<topic>.md` files (matmul, ccl,
   kernels, models, operators, sharding, ...) for content relevant to the subject.
   These provide starting pointers but are not required — the skill works without
   them.

3. **Dispatch research subagent**: Launch an Agent with `research-prompt.md` as
   instructions. Pass the subject, any matched reference content, and the
   refresh flag. The subagent does the actual Grep/Read/deepwiki work and
   produces the entry body.

4. **Write entry**: Invoke `/tt:note` with topic=`learn-<subject-slug>`,
   title=<one-line summary>, body=<the research entry produced in step 3>.

5. **Return entry**: Return the entry body so the caller can use it immediately.

## Entry body convention

```markdown
## <one-line summary of the research subject>
**<timestamp>** · `<source-repo>@<short-sha>`

**Core insight:** <1–3 sentences: the single most important thing to know.>

**How it works:**
- <Concise bullet points — only what's needed to act on the subject>

**Key files:**
- `path/to/file` — one-line description
```

**Body target: under 80 lines.** Entries become part of agent context — every
line costs.

## Refresh

Entries are assumed fresh for a development session. To force re-research:
- User says "refresh" or "re-learn"
- Caller passes a refresh hint
- This skips step 1 and always researches from scratch, writing a new entry on
  top of the existing timeline.

## Failure Mode

If neither local search nor deepwiki produces useful results, write a note
documenting what was attempted, what references were checked, and what's missing.
Escalate to the user — don't fabricate understanding.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Research subagent instructions | `research-prompt.md` |
