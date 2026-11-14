# CLEAR-EC KG — operating instructions

An LLM-maintained knowledge graph for the CLEAR-EC Grand Challenge (predict CD, CV and HEX from corneal endothelium
specular images; score = mean of the per-metric MAPE). It links every experiment, finding, idea, decision, PI rule and
paper, so that a new idea is checked against what was already tried and against the literature before GPU time is
spent. Layout copied and adapted from `/Users/byounghanlee/EHR-KG` (2026-11-08). All text is English.

Read `purpose.md` first (questions and current stance), then this file.

## Settings

| Name | Value |
|---|---|
| `KG` | `/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research/kg` (the path has a space: always quote it) |
| `PY` | `/usr/bin/python3` (has PyYAML; the scripts re-exec themselves with it if another interpreter lacks PyYAML) |
| Pod | `clear-ec-research-4gpu-1104`, root `/workspace/clear-ec-research/experiments` (read only from KG sessions) |

```bash
KG="/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research/kg"; PY=/usr/bin/python3
$PY "$KG/scripts/build_graph.py"            # after any note change; must end with errors={}
$PY "$KG/scripts/build_viewer.py"           # then refresh viewer/index.html (self-contained; do not publish)
$PY "$KG/scripts/kg_query.py" check mixup    # the idea check (see below)
```

## Layout

- `purpose.md` — the questions this KG answers and the current stance (lane bests with baselines and folds).
- `schema/edges.yaml` — node types, required fields, controlled values, edge vocabulary. Read it before adding a field or edge.
- `sources.yaml` — what was ingested and how far each lane `RESULTS.tsv` was read (`reviewed_through`).
- `wiki/` — one Markdown note per node, YAML frontmatter + body (Obsidian-compatible):
  `papers/ methods/ datasets/ experiments/ findings/ ideas/ decisions/ constraints/ syntheses/`.
  `wiki/_templates/` has one template per type. `wiki/index.md` and `wiki/_views/*.md` are generated: never edit.
  `wiki/log.md` is an append-only operation log.
- `graph/` — generated: `graph.json`, `lint.md` (errors, warnings, info, orphans), `references.bib`.
- `viewer/index.html` — generated, self-contained explorer (no network): map coloured by node type, search by id or
  text, click a node for its summary, fields and typed edges in and out. Rebuild with `build_viewer.py` after every
  `build_graph.py`. Local file only: do not publish it.
- `scripts/` — `build_graph.py` (compile + views + lint), `kg_query.py` (shell queries), `check_notes.py` (validate
  given notes; safe while others edit), `unrecorded_rows.py` (lane rows not yet ingested), `kglib.py` (shared),
  `build_viewer.py` + `graph_layout.py` + `viewer_template.html` (the explorer, ported from EHR-KG on 2026-11-09).

## Ownership

- The paper agent writes only `wiki/papers/*.md`, from `wiki/_templates/paper.md` or `paper-light.md` (the EHR-KG
  paper template; `relevance_circe` / `relevance_clearec` both mean relevance to CLEAR-EC). Do not edit paper notes
  from other sessions; add links from your own node instead (`literature`, `builds_on`, `motivated_by`).
- Everything else in `kg/` belongs to the KG maintainer (this file's operations).
- Links from papers to project nodes are written by the maintainer on the project node, never in the paper note:
  `literature` edges with a `relation` from `schema/edges.yaml` `literature_relations` (corroborates, contradicts,
  suggests, source-of-method, source, source-of-data, explains, tests, closes) and a `note` saying which part of the
  paper's suggests_try / relevance_note / key_numbers it rests on. Papers marked `superseded` (for CV/HEX) link to
  `dec-014` and the root-cause findings with relation `closes`; only their non-superseded parts get other links.
- Paper notes are accepted as written: free-text values such as `uses_method: [Dense U-Net]` are ignored; only
  `[[wikilinks]]`, prefixed ids (`method-`, `ds-`, `idea-`, `find-`, `dec-`, ...) and citekeys count as links. A paper
  link to an id with no note becomes a stub node and a lint warning, never a build error.

## Node types and ids

| Type | Folder | Id | Required beyond id/type/name/summary |
|---|---|---|---|
| paper | papers | citekey `lastnameYEARword` | title (authors, year, tldr recommended) |
| method | methods | `method-<name>` | — |
| dataset | datasets | `ds-<name>` | — |
| experiment | experiments | `exp-<lane>-<name>` (`exp-cd-`, `exp-cv-`, `exp-hex-`, `exp-bb-`, `exp-deploy-`, `exp-rc-`, `exp-gpu0..3-`, `exp-hist-`, `exp-ref-`) | lane, date, status, direction, metrics, folds, baseline, value, delta, source |
| finding | findings | `find-<name>` | status, source |
| idea | ideas | `idea-<name>` | status, origin |
| decision | decisions | `dec-NNN-<name>` | status, date |
| constraint | constraints | `con-<name>` | status, origin (`user` = the PI), source; `scope: global` for rules binding all work |

Key edges: experiment `uses_method`, `baseline_ref`, `stacked_on`, `superseded_by`, `supports_finding`,
`contradicts_finding`; idea `tested_by` (with `closeness: same | close | related` and a `note` on what differs),
`motivated_by`, `builds_on`, `literature`; paper `evidence_for` / `evidence_against` (to `idea-`, `dec-`, `find-`);
constraint `constrains`; decision `decides`, `supersedes`; finding `qualifies` (limits another finding without replacing
it) and `contradicts` (evidence that an assumption of a finding, idea or decision is wrong; say which in `note`). Any node may point to a paper with `literature`
(`relation: corroborates | contradicts | suggests | source-of-method | explains`).

## Experiment nodes: the rules

- **Baseline and folds are mandatory**, in the same node as the number. The baseline names what the delta is against
  and its value on the same rows (e.g. "best pipeline CD 6.264, folds 0-4" or "CD stack 6.204"). Deltas against
  different baselines never add up; say which one a sentence uses. A coordinator error came from quoting ViT-L's
  -0.036 (against the best pipeline) as a gain on top of the 6.204 stack.
- Record numbers **as written** in the source. If a number, baseline or fold set is unclear, keep it as written, set
  `ambiguous: true` and explain in `ambiguous_note`. Never guess. Arithmetic you did yourself is labelled "(arithmetic)".
- `delta_num` is the number against `baseline` (negative = better) for sorting; leave it empty when unclear.
- `status`: kept | dropped (one version failed, direction open) | running (incl. queued) | closed (direction closed as
  tested) | superseded (replaced by a later read; set `superseded_by`) | reference (baseline read) | measurement.
- `direction`: tries of one direction share the exact same string; the closed view groups by it.
- `source`: file and line, e.g. `experiments/lane_cd_20261108/RESULTS.tsv line 51 (snapshot 2026-11-09 13:17 UTC)`.
  One node per meaningful try: sub-reads (partial folds, timing lines, repeated HEX chain re-runs) go into the
  node's body or source, not into new nodes.

## Operations

### Ingest new lane results
1. Copy the file read-only (never write on the pod, never `du` on `/workspace`):
   `perl -e 'alarm 90; exec @ARGV' kubectl exec clear-ec-research-4gpu-1104 -- bash -c 'cd /workspace/clear-ec-research/experiments && cat lane_cd_20261108/RESULTS.tsv' > /tmp/cd.tsv`
   Also read the lane `REPORT.md` for statuses (continue/stop, closed lists).
2. `$PY "$KG/scripts/unrecorded_rows.py" lane_cd_20261108 /tmp/cd.tsv` lists rows after `reviewed_through`.
3. For each meaningful try: write `wiki/experiments/exp-<lane>-<name>.md` from the template, or update the existing
   node of that direction (new value, status, source line). Link `uses_method` (create the method if missing),
   `stacked_on` for kept steps, `supports_finding` when it backs a finding, and add the experiment to the
   `tested_by` of any idea it tests. When a direction is closed (LANES closed list or the lane writes stop), set
   every try in it to `closed`.
4. Update `purpose.md` "Current stance" if a lane best changed, raise `reviewed_through` in `sources.yaml`.
5. `$PY "$KG/scripts/check_notes.py" <changed notes>`, then `$PY "$KG/scripts/build_graph.py"` (errors={}),
   then append a line to `wiki/log.md`: `## [YYYY-MM-DD] ingest | <what>`.

### Ingest a paper (paper agent)
Write `wiki/papers/<citekey>.md` from the paper template. Link to project nodes with `evidence_for` /
`evidence_against` (ideas, decisions, findings) and `supports_finding`; use only ids that exist
(`kg_query.py find <word>`). Run `check_notes.py` on the note; warnings about unknown targets are allowed.

### Capture a finding, idea, decision or constraint
Use the template. A finding states its number with baseline and folds and cites the experiments or document behind
it. An idea gets `motivated_by`, `tested_by` (closest tries, with closeness) and a dated `novelty_check`.
A constraint quotes the PI only for the PI's own words (`origin: user`); coordinator rules are `origin: claude`.
Decisions are never deleted: a replacement sets `supersedes` and the old one becomes `superseded`.

### Query
`kg_query.py` reads `graph/graph.json` (rebuild first if notes changed):
```bash
$PY "$KG/scripts/kg_query.py" check <words...>     # all nodes (incl. note bodies) matching every word, with numbers
$PY "$KG/scripts/kg_query.py" closed [--lane cv] [--metric CV]   # closed directions with best try, baseline, folds
$PY "$KG/scripts/kg_query.py" kept [--lane hex]    # kept steps, running work, references
$PY "$KG/scripts/kg_query.py" experiments [--lane L] [--status S] [--metric M]
$PY "$KG/scripts/kg_query.py" node <id> [--all]    # fields and typed edges in/out
$PY "$KG/scripts/kg_query.py" ideas [--status never-tried] | findings | constraints | papers
$PY "$KG/scripts/kg_query.py" edges tested_by [--to exp-...] [--from idea-...]
```
Views (regenerated by every build): `wiki/_views/tried-and-closed.md`, `kept-stack.md`, `experiments-by-lane.md`,
`open-ideas.md`, `open-ideas-ranked.md`, `findings-list.md`, `constraint-list.md`, `decision-log.md`, `papers-table.md`.
`open-ideas-ranked.md` orders open ideas by the share of the projected-test gap to first place they could close
(gaps in `schema/edges.yaml` `ranking`); it reads the idea fields `owner_lane`, `cost`, `cost_hours`, `cost_unit`,
`potential_metric` (CD | CV | HEX | all) and `potential_gain` + `potential_basis`. The gain is an estimate: write its
basis (closest try's delta, variance share, paper effect size) and say "speculative" when OOF cannot measure it.
A reusable answer goes into `wiki/syntheses/syn-<topic>.md`.

### Lint
`build_graph.py` writes `graph/lint.md`. Errors (broken links from non-paper notes, YAML errors, duplicate ids,
id/filename mismatch, bad controlled values, edge-type mismatch) must be fixed. Warnings: unresolved paper links,
experiments without a method, ideas without motivation or with a tested status but no `tested_by`, constraints
without targets (unless `scope: global`). Info: ambiguous numbers (`ambiguous: true`), ideas without a literature check.

### Daily re-ingest (once a day, after the lanes' 09:00 Eastern continue/stop review; ~20 min)
1. Copy each lane's `RESULTS.tsv` and `REPORT.md` read-only to a scratch folder (one `kubectl exec ... cat` per file
   under `perl -e 'alarm 90; exec @ARGV'`; never write on the pod, never `du` on `/workspace`). Sources: the four lane
   folders in `sources.yaml` `lanes` (cd, cv, hex, backbone) plus any new analysis folder the coordinator names
   (e.g. `rootcause_YYYYMMDD/out`). Diff each `REPORT.md` against the previous copy for status changes.
2. `$PY "$KG/scripts/unrecorded_rows.py" <lane folder> <copy>` for each lane: rows after `reviewed_through`.
3. Ingest as in "Ingest new lane results": one node per meaningful try (new node, or update the direction's node with
   the new value, status and source line, keeping earlier reads under `## Versions`); set `closed` when the lane writes
   stop; add the experiment to the `tested_by` of any idea it tests; update the idea's `status`, `cost` and
   `potential_*` if the read changes them. Resolve an `ambiguous` note only when a source now states the missing
   number, fold set or baseline; otherwise append what was re-checked to `ambiguous_note`.
4. Update `purpose.md` "Current stance" (snapshot time, lane bests with baseline and folds) and raise
   `reviewed_through` / `snapshot` in `sources.yaml`.
5. `$PY "$KG/scripts/check_notes.py" <changed notes>`, then `$PY "$KG/scripts/build_graph.py"` (errors={}), then
   `$PY "$KG/scripts/build_viewer.py"`. Read `graph/lint.md` (warnings, ambiguous count, orphans).
6. Append one line to `wiki/log.md`:
   `## [YYYY-MM-DD] ingest | daily re-ingest, snapshot HH:MM UTC: cd lines A-B, cv C-D, hex E-F, backbone G-H; N new
   experiments, M updated (ids); ideas changed (ids); ambiguous X -> Y; build nodes=..., edges=..., errors none`.

## How a lane session checks an idea (do this before spending GPU time)

1. **Search the KG.**
   ```bash
   KG="/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research/kg"
   /usr/bin/python3 "$KG/scripts/build_graph.py" >/dev/null
   /usr/bin/python3 "$KG/scripts/kg_query.py" check <2-3 distinctive words>      # e.g. check window crop ; check mixup
   grep -ril "<word>" "$KG/wiki" | grep -v _views                              # catches wording the check missed
   ```
   Then read `wiki/_views/tried-and-closed.md` for your metric and `kg_query.py constraints`. For each hit, compare
   the **baseline and folds**, not the bare number. "Closed" means closed as tested: the same test on the same
   inputs is not worth rerunning; a clearly different version is fine.
2. **Quick literature search** (a few minutes): `kg_query.py papers` and `check` cover papers already in the KG; then
   one or two web searches (e.g. "specular microscopy endothelium <idea>", "cell segmentation <idea>",
   "MAPE regression <idea>"). Do not download files.
3. **Record the link.** If the idea is new, write `wiki/ideas/idea-<name>.md` with `motivated_by`, `tested_by`
   (closest tries and what differs), `literature` (papers found) and a dated `novelty_check` naming the searches.
   If it repeats a try, add the closeness note to that idea instead of running it. When the run finishes, ingest the
   result as an experiment and add it to the idea's `tested_by`. Rebuild and log.

## Rules

- Numbers only from sources, as written, with baseline and folds; mark ambiguity; never invent.
- Compare with Phase II only (submission 1 test 8.9675 / OOF 8.877, or the best pipeline OOF 8.739); Phase I is legacy.
- Read-only on the pod; no downloads; no git commits; no messages to other sessions from KG maintenance.
- Do not edit `wiki/index.md`, `wiki/_views/` or `graph/`; do not edit paper notes you did not write.
- Times in US Eastern unless a source writes UTC (then say UTC).
