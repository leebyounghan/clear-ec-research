# E11 — push the shipped pre-only detector without challenge labels: FAIL (both arms)

Pre-registered gate (protocol.json): all-25 miss <= 0.12 or jitter <= 0.19, with FP <= 0.18; and val-5 miss/FP/jitter each within +0.02 of the shipped detector. Scoring = E10 `score25` (25 expert-annotated images, hull region dilated 0.5 spacing, match 0.5 x spacing, F1 at 12 px). Reference = shipped pre-only `sg2pre_s46`.

| detector | all-25 miss | FP | jitter | F1 | val-5 miss | FP | jitter | F1 | gate |
|---|---|---|---|---|---|---|---|---|---|
| shipped pre-only s46 | 0.150 | 0.165 | 0.215 | 0.862 | 0.170 | 0.192 | 0.214 | 0.857 | reference |
| E11a noisy student | 0.196 | 0.095 | 0.214 | 0.859 | 0.207 | 0.125 | 0.220 | 0.848 | FAIL (miss +0.046; val-5 miss +0.037 > 0.02) |
| E11b SREP + synthetic v5 mixed | 0.101 | 0.264 | 0.253 | 0.865 | 0.148 | 0.360 | 0.282 | 0.826 | FAIL (FP 0.264 > 0.18; val-5 FP +0.17, jitter +0.07) |

## E11a — noisy student from the teacher's own peaks (real frames outside fold 4)
- Data: teacher (shipped) peaks on 7164 real frames (store threshold 0.2, median 950 peaks per frame, median 544 positives >= 0.6); 6875 frames pass the >= 100-positive gate. 4000 updates, batch 2, photometric + flip augmentation, encoder lr 1e-5 / head 3e-4; final loss 0.229 (1514 s on one H200).
- Effect: the student becomes more conservative. FP falls on every image (mean -0.070, max -0.149) while misses rise on 22/25 (mean +0.045, up to +0.112 on 0334-23 and 0585-23, the two hardest images). Jitter unchanged (-0.001). Detections per image 128 vs 144 (ground truth 142). F1 unchanged (-0.003).
- Reading: self-training on the teacher's confident peaks reinforces the confident cells and suppresses the uncertain ones; recall, which is the HEX bottleneck (E4 break-even miss ~8 %), moves the wrong way.

## E11b — SREP + synthetic v5 mixed training from the fold-4 encoder
- Data: SREP 307 Topcon images (hull-masked targets) + 3000 synthetic v5 images (one per update alongside an SREP batch of 2), 6000 updates, head lr 1e-3; final loss SREP 0.399 / synthetic 0.603 (1882 s).
- Effect: recall improves (miss -0.049 mean, 19/25 images better, passes the 0.12 rule) but the synthetic FP/jitter profile carries over: FP rises on 18/25 (mean +0.099; 0326-23 0.14 -> 0.79, 0365-23 0.40 -> 0.79) and jitter rises on 23/25 (+0.038). Detections per image 163 vs 144. F1 +0.003.
- Reading: the synthetic set teaches the detector to fire in low-contrast and merged regions where the real images have no cell, the same failure as the synthetic-only E10 detector (FP 0.542), diluted but not removed by the SREP half.

## Conclusion
Neither arm moves the detector toward the HEX requirement (recall >= 92 % at < 4 px jitter). The two arms fail in opposite directions (student: precision for recall; mixed: recall for precision and jitter), which is the trade-off already visible along the shipped detector's threshold axis (E7). No label-free route (A verifier, B synthetic-only, E11a/E11b) passes a gate; the shipped hex6 blend (-0.0311 pp mean on B) remains the only green component.

## Run record
- First attempt (pod clear-ec-research-4gpu-1026) died 2026-11-04 05:44 UTC: cgroup OOM inside `load_pseudo()` (per-row `z['peaks']` re-reads of the npz kept a full copy per row, ~139 GB per file). Fixed 13:1x UTC (read each member once; dataloader workers 2/2). Rerun on clear-ec-research-4gpu-1104 (192 GiB): both trainings concurrently at 6.6 GB cgroup memory; finished 13:42 / 13:48 UTC.
- Files: score25_student.json, score25_mixed.json, train_student.json, train_mixed.json, detector_student.pt, detector_mixed.pt (pod only), logs.
