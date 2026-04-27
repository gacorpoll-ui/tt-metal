# Note-writing protocol

Canonical convention for writing entries to `~/.tt-agent/notes/`. Loaded by
`tt:note`'s pipeline. Other skills do not load this file directly — they
invoke `/tt:note`.

## Filename rules

`~/.tt-agent/notes/` is a flat directory — no subdirectories. One markdown file
per topic.

| Topic kind | Pattern |
|---|---|
| Workflow scope (optimizer / debugger / tester) | `<scope>.md` |
| Orchestrator task | `<task-slug>.md` |
| Research subject (tt-learn) | `learn-<subject>.md` |
| Code review log | `code-review.md` |

Topic-slug collisions across source repos are resolved by picking a more
specific slug at write time (e.g., suffix the slug with the source-repo
basename).

## Topic file shape

```
# <topic>

## <newest-entry-title>
**<timestamp>** · `<source-repo>@<short-sha>`

<body>

## <next-newer-entry-title>
**<timestamp>** · `<source-repo>@<short-sha>`

<body>
```

- Single `# H1` — the topic name. Written at file creation; never modified.
- Entries are H2 sections, **prepended** under the H1 (newest-on-top).
- The latest entry carries the live-state snapshot; older entries preserve history.

## Entry shape

- **Title (H2)** — punchy summary. Mirrors the commit subject.
- **Metadata line** — `**YYYY-MM-DD HH:MM**` + `` `<source-repo>@<short-sha>` ``.
- **Body** — free-form. Sub-section structure is the calling skill's choice.

### Source-SHA capture

`<source-repo>@<short-sha>` reflects the active source-code workspace at write
time:

- `<source-repo>`: from `git -C <workspace> remote get-url origin`, basename only (e.g., `tt-metal`).
- `<short-sha>`: `git -C <workspace> rev-parse --short HEAD`.
- If `git -C <workspace> status --porcelain` is non-empty, append `-dirty` (e.g., `<source-repo>@<short-sha>-dirty`).

## Commit model

One commit per phase / per attempt / per "anything worth noting." Skills
decide when a phase boundary has been reached. No auto-commit-on-write.

### Subject format

```
<topic>: <entry-title>
```

The H2 entry title prefixed by the topic slug. Commit body is usually empty
— the diff carries the entry content.

## Auto-init

On first write, if `~/.tt-agent/notes/.git/` is absent:

```bash
cd ~/.tt-agent/notes
git init
git add -A && git commit -m "init: capture existing notes"
```

Idempotent — subsequent writes detect `.git/` and skip.

## Atomic-write protocol

One entry per file per invocation. `git add <file> && git commit -m "<subject>"`
as one bash compound.

**Per write:**

1. `BEFORE_HEAD=$(git -C ~/.tt-agent/notes rev-parse HEAD 2>/dev/null || echo "")`
2. Read the topic file (initialize with `# <topic>\n\n` if absent).
3. Prepend the new entry; write the file.
4. `git -C ~/.tt-agent/notes add <file> && git -C ~/.tt-agent/notes commit -m "<subject>"`.
5. Verify `git -C ~/.tt-agent/notes rev-parse HEAD~1` equals `BEFORE_HEAD`. If
   not, another commit landed during 2–4. Run `git reset --soft HEAD~1`,
   re-read from new HEAD, re-prepend, retry step 4 once.

If `BEFORE_HEAD` is empty, perform Auto-init first, then proceed at step 1.

## Cross-topic referencing

When one skill invokes another and both produce notes, each is its own
`/tt:note` invocation → its own commit. Express the relationship as pointer
text in the entry body:

```markdown
## <pointer-entry title>
**<YYYY-MM-DD HH:MM>** · `<source-repo>@<short-sha>`

See `<other-topic>.md` (entry written this timestamp).
**TL;DR:** <1–2 sentence summary of the referenced entry>.
```

Readers find the target via `git log --oneline -- <other-topic>.md`.

## Operational policies

- **Sync:** local-only by default; standard git remotes for sharing.
- **Concurrency:** no explicit locks; same-file race handled in § Atomic-write protocol.
- **Ad-hoc human entries:** humans edit + commit directly following this convention.
- **Pruning:** none automatic; developer-driven when a file becomes unwieldy.
