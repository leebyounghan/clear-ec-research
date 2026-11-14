---
id: dec-021-gpu-shutdown-closeout-20261113
type: decision
name: GPU shutdown and close-out order of 2026-11-13 (D-GPU-002)
summary: 'PI order, 21:40 EST 11-13: take everything down on GPUs 0, 1, 2 and 3. Done at 21:41 EST by pid after checking each command line; all four cards read 0 %, 0 MiB. No lane starts a keepalive or any GPU job from here on, which supersedes the keep-busy rule for this pod. The PI also ordered the close-out: every lane writes a final summary, the coordinator merges them, unneeded sessions and goals are removed, and the research tree is cleaned with models kept. The no-GPU-job part was lifted the same evening for a CD training-setting screen (ledger line 211).'
status: accepted
date: '2026-11-13'
deciders:
- PI
source: shared notes ledger_20261111.jsonl D-GPU-002 (line 206; supersedes D-GPU-001, line 182), D-CD-TRAIN-SCREEN (line 211); docs/FINAL_SUMMARY_20261113.md sections 0, 4 and 6
decides:
- '[[con-no-gpu-jobs-after-20261113-shutdown]]'
- '[[con-gpu-not-idle]]'
motivated_by:
- '[[exp-deploy-t4-all-fp32-200-rows]]'
---

# GPU shutdown and close-out order of 2026-11-13 (D-GPU-002)

PI order, 21:40 EST 11-13: take everything down on GPUs 0, 1, 2 and 3. Done at 21:41 EST by pid after checking each command line; all four cards read 0 %, 0 MiB. No lane starts a keepalive or any GPU job from here on, which supersedes the keep-busy rule for this pod. The PI also ordered the close-out: every lane writes a final summary, the coordinator merges them, unneeded sessions and goals are removed, and the research tree is cleaned with models kept. The no-GPU-job part was lifted the same evening for a CD training-setting screen (ledger line 211).

## Context

The last GPU work was the coordinator's mixed-precision read on GPUs 1-3 and yielding keepalives on idle cards (O-CD-GPU0-KEEPALIVE, O-BB-GPU13-KEEPALIVE). D-GPU-001 (coordinator, 11-12 22:40 EST) had given GPUs 0, 1 and 3 to the deploy lane for the 200-row reads.

## Decision

Stopped: the CD lane's keepalive on GPU 0 (pid 643028) and the coordinator's mixd runs on GPUs 1-3 (pids 691335, 691212, 690742; unfinished at 90 of 200 rows and not needed, because the all-fp32 read had already answered the question). No lane starts a keepalive or any GPU job afterwards ([[con-no-gpu-jobs-after-20261113-shutdown]]); for this pod that replaces the standing keep-busy rule ([[con-gpu-not-idle]]) by the PI's own instruction.

## Consequences

- Close-out order: five lane final summaries (CD, CV, HEX, backbone, deploy) and one consolidated summary (`docs/FINAL_SUMMARY_20261113.md`, 22:15 EST), then removal of unneeded sessions and goals, then a clean-up of the research tree organised as data / models / ideas, with the KG updated.
- Clean-up plan as written in the consolidated summary (section 4): models are kept; nothing a rebuild of p / t3 needs is touched before the submission is final; member prediction files are kept wherever checkpoints go; by-name references are checked with grep before any folder is deleted. Classes: DELETE-NOW (scratch, partial, stale), DELETE-AFTER-SUBMISSION, ARCHIVE, and folders with no owner yet. Resolved conflicts: `lane_hex_20261108/seg_loose` stays (the CD stack reads it), `submission1_test_20261030` and `lane_gpu2_20261106` stay (rebuild closure).
- This KG ingest deleted nothing anywhere.
- Five analysis reports were being written by separate agents when the summary closed (hyperparameters, data, post-processing, bias, CD literature); they are not ingested.

## Lifted the same evening

Ledger line 211 (D-CD-TRAIN-SCREEN, stamped 22:36 EST): a DINOv3-B/16 fold-0 training-setting screen was launched on GPUs 0-3 at 22:34 EST, '(PI via coordinator 22:50 EST, D-GPU-002 lifted)'; the time 22:50 is as written and is later than the line's own stamp. So the no-GPU-job rule held for about an hour ([[con-no-gpu-jobs-after-20261113-shutdown]], status lifted; [[exp-cd-training-setting-screen-dinov3]]). A second launch followed at 22:43 EST: two SSL arms on GPUs 0 and 1 ([[exp-cd-ssl-pretrain-screen-dinov3]]). The PI also asked for further CPU reads on CD after the summary closed ([[exp-cd-vitl-averaged-tracks-in-stack]], [[exp-cd-recipe-a-e9-12-window-slot]]) and closed a fellow-eye transfer as a trick ([[exp-cd-fellow-eye-residual-transfer]]). The consolidated summary's statement that no lane process is running (22:15 EST) predates these lines.

## Revisit when

The PI decides on the DELETE-AFTER-SUBMISSION class (the PI's instruction so far is to keep models), or the running screen changes the CD model.
