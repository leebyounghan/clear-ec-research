# Framework revision, 2026-11-11 (coordinator's proposal)

Source: Wu and Liu, "SciMARS: Multi-Agent Recursive Self-Improvement for Closed-Loop
Scientific Discovery", arXiv:2611.12552v1 (the PI asked for the project framework to be
revised against it). Everything below is the coordinator's reading and implementation.
None of it is a PI rule until the PI says so. `LANES_20261108.md` stays the charter and is
not edited by this file.

## What the paper says, in one paragraph

A fixed multi-agent team fails in two ways. Evidence stays in one agent's history, so
others repeat work or miss constraints. And fixed role procedures, hand-offs and memory
rules keep repeating a mistake even after the evidence against it has been recorded. The
paper's answer has four parts: a shared, typed, append-only evidence graph; separate
editors that change one thing at a time (role procedures, one hand-off contract, memory
policy); probes that test a changed team before it replaces the current one, with a
matched comparison required to replace the incumbent; and a reflection step that writes
short lessons (cue, lesson, evidence, limits, when to reconsider).

## Where our framework showed the same failures (11-10 afternoon to 11-11 midday)

| Paper's failure mode | What happened here |
|---|---|
| Recorded finding not acted on | "A member that is better alone gets weight 0 in the stack" was observed nine times, and the coordinator still launched five more member-additive experiments. Same for "a folds 0+4 gain shrinks on five folds" and "a single-seed gain is seed luck". |
| Constraint known to one agent does not reach the others | The CV lane found the inner-stretch leak (0.006) at 10:00; the HEX chain carried a 0.010 selection optimism that nobody had measured; "deploy fit must be the evaluated procedure" was only stated after the HEX joint ridge was adopted. |
| Hand-off drops or reorders | Three coordinator messages crossed with HEX lane reports within 40 minutes; the HEX freeze candidate changed chain → ridge → chain → ridge. Lanes hit the app's 10-message limit and could not report to each other. |
| Lost observations | The size/runtime table and a lane's notes file could not be found by the coordinator; the KG has not been re-ingested since 11-09, so "closed" lists are stale. |
| Verifier says yes to a wrong artefact | Synthetic frames v1 had inverted polarity and passed the lane's own check; a lane's "correction" compared two numbers computed under different conventions. |
| Fixed roles | Lanes are fixed per metric. The CD lane had no experiment for 14 hours; nobody owned strict verification or the integrated table; deploy work landed on the cell-route lane by accident. |

## Changes (each is one small edit, as in the paper)

### 1. Shared evidence ledger (memory)

- File: `<shared notes>/ledger_20261111.jsonl`, append-only, one JSON object per line.
  Shared notes folder:
  `~/Library/Application Support/Claude/side-session-notes/local_5a5ca17b-018d-4ccd-b356-f8794b094404/`.
- Record fields: `id`, `type`, `author`, `time_est`, `text` (one or two sentences),
  `value` (numbers with folds and baseline named), `source` (file path on the pod or in
  the repo), `links` (list of `{rel, id}` with rel in supports / contradicts /
  derives_from / depends_on / supersedes).
- Types: `observation` (a measured number), `finding` (what it means), `decision`
  (coordinator or PI), `lesson` (cue, lesson, reconsider-when), `question` (open),
  `artifact` (a file other lanes need: path, sha256 if weights).
- Nothing is edited or deleted. A correction is a new record that `supersedes` the old one.
- Every lane reads the records added since its last turn before acting, and appends its
  own results there in addition to its `RESULTS.tsv`. The ledger is also the transport when
  the app's message limit blocks a lane.
- The coordinator ingests the ledger into the KG; lanes still do not edit the KG.

### 2. Versioned decisions (hand-off contract)

- Every coordinator decision carries an id, e.g. `D-HEX-004`, and names what it
  supersedes. Lanes act on the highest id for their topic only and quote the id in the
  report. A lane report that would change a decision names the decision id it answers.
- A lane never reverts a freeze file on a message that carries no decision id.
- One open decision per topic: the coordinator does not send a second instruction on a
  topic while a lane's check for the first is running.

### 3. Lessons as a pre-launch gate (reflection)

Before the coordinator or a lane starts a GPU job for a new direction, it writes one line
answering each active lesson's cue. Active lessons on 11-11:

- `L1` Member alone vs stack. Cue: the proposal is a new image-to-label member or a
  variant of one. Lesson: single-member gains of 0.01 to 0.07 have given stack weight 0
  ten times. Gate: state what information the stack lacks that this adds; run the stack
  read on folds 0+4 before any further fold. Reconsider when a member's residual
  correlation with the stack is below 0.9.
- `L2` Screen folds. Cue: a folds 0+4 gain. Lesson: it shrinks or vanishes on five folds
  (seven times). Gate: never quote it as a gain; five folds decide.
- `L3` One seed. Cue: a gain that rests on one trained seed. Lesson: order-loss DINOv3 was
  -0.0460 with one seed and -0.0369 with three. Gate: read with the seed mean before
  adoption.
- `L4` Nesting. Cue: any stack, chain or selection number. Lesson: selection on reported
  rows was worth 0.006 (CV stretch), 0.010 (HEX chain) and 0.010 (cell-route version
  choice). Gate: quote the strictly nested value; the as-computed value goes next to it.
- `L5` Deploy equals evaluated. Cue: a new stack form. Lesson: the HEX ridge's nested value
  was 9.7266 under one convention and 9.7414 under the one its deploy fit used. Gate: the
  nested code run with no outer fold must reproduce the deploy parameters.
- `L6` Like for like. Cue: comparing two numbers. Lesson: a lane's correction compared a
  deploy-convention number with a chain-convention number. Gate: name the convention,
  folds and baseline of both numbers in the same sentence.
- `L7` Effect vs noise. Cue: adopting or dropping on a difference under 0.005. Lesson:
  seed noise is about ±0.005 and leave-one-out cascades move 0.001. Gate: pre-register the
  rule before the read; no cascades.

### 4. Roles (one change)

- The CD lane becomes the **verifier** for the integrated candidate until the freeze: it
  recomputes the three-metric table strictly nested from the lanes' files, checks that each
  lane's deploy parameters are the evaluated procedure with no outer fold, and reports
  disagreements. It did not author the CV or HEX stacks, which is the point.
- The cell-route lane is named what it already is: the **deploy** lane.
- The backbone lane stays the one **exploration** lane. Metric lanes stop proposing
  member-additive experiments (lesson L1).

### 5. Probes before commitment

- A direction gets its cheapest killing test first (a CPU read on existing predictions, a
  zero-shot read, a 200-frame sanity check, a picture next to real data), and the
  coordinator looks at the artefact itself when it is an image or a table.
- Replacing an incumbent (a freeze component) needs a matched comparison: same folds, same
  nesting, same convention, incumbent recomputed in the same run.

## What is not taken from the paper

- No automatic editing of lane prompts or roles by an agent; the coordinator proposes and
  the PI decides.
- No per-round re-selection of the whole team; three days remain.
- The paper's results come from benchmark replays with small counts (for example 12 of 18
  against 8 of 18); they motivate these changes but do not measure what they are worth here.

## Status

- Applied 2026-11-11 by the coordinator: sections 1, 2, 3 and 5, and the verifier
  assignment in section 4, as working practice for the remaining days.
- 2026-11-11 afternoon: the PI approved adding these to the charter. They are now the
  section "Shared ledger, decision ids, lessons, verifier" in `LANES_20261108.md`.
