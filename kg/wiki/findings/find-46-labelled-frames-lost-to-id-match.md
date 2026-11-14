---
id: find-46-labelled-frames-lost-to-id-match
type: finding
name: '46 labelled, readable frames never reached the whole-frame models: an exact-string ID match dropped them (9,000 downloaded, 8,954 cached)'
summary: The download has 9,000 frames and 9,000 label rows; the cache and the whole-frame memmap have 8,954. The 46 missing frames are labelled and readable and were lost to an exact-string ID match (trailing space 33, hyphen / space 10, other spelling 3; all new-format IDs). They are not in the whole-frame models and are in the Phase-1 crop members (8,998-row full-data fit).
status: robust
metrics: []
date: '2026-11-13'
source: docs/FINAL_SUMMARY_20261113.md section 2, 'Row accounting (verified 11-13)' (revised 22:19 EST 11-13; taken there from `analysis_data_20261113.md`, which is not ingested)
motivated_by:
- '[[find-two-sources-id-formats]]'
---

# 46 labelled, readable frames never reached the whole-frame models: an exact-string ID match dropped them (9,000 downloaded, 8,954 cached)

The download has 9,000 frames and 9,000 label rows; the cache and the whole-frame memmap have 8,954. The 46 missing frames are labelled and readable and were lost to an exact-string ID match (trailing space 33, hyphen / space 10, other spelling 3; all new-format IDs). They are not in the whole-frame models and are in the Phase-1 crop members (8,998-row full-data fit).

## Evidence

Row accounting as written in the consolidated summary: download 9,000; cache and whole-frame memmap 8,954 (`code/cache_native.py:7`); scored CD, CV 8,954 and HEX 8,952 (two HEX-zero rows, `2019-0064-ODCN`, `-OSCN`); full-data fit of the whole-frame models 8,952; full-data fit of the Phase-1 crop members 8,998 (from the reconciled cache `datasets/clear_ec_final9000_20260904/`); fold fits of the whole-frame models 7,150-7,161 per fold (0-12 rows per fold dropped because they are linked to validation by pixel-identical frames; 14 pixel-identical pairs exist under different IDs); stack and stretch fits 8,934 (the 20 detector-training frames left out); cell-route ridge fit 8,825. New-format rows: 4,440 in the cache against 4,486 in the CSV. The summary adds that '8,953' in BRIEF and in the KG dataset node matches no file (a prose error).

## Limits

Recorded from the consolidated summary's table, which calls it verified on 11-13; the data analysis report behind it was not read by the KG ingest, and no experiment node measures what the 46 frames would be worth.

## Bears on

No model was retrained with the 46 frames; the sources record no decision about them. Dataset note: [[ds-clear-ec-train-8953]].
