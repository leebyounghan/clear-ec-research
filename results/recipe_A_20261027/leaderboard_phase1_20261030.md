# CLEAR-EC Phase I leaderboard (read 2026-10-30 17:45 UTC, public page; best published result per participant)

Score = equal-weight mean of per-image absolute percent errors for CD, CV, HEX (lower is better); 100 hidden test images.

| # | User | Algorithm | Date | Score |
|---|---|---|---|---:|
| 1 | abdulsohu | V14 | 14 Sep 2026 | **8.6206** |
| 2 | Leebyounghan | Lamda&KIM | 15 Sep 2026 | **8.7010** (ours) |
| 3 | raihan | ec blend | 14 Sep 2026 | 8.7345 |
| 4 | nancyibrahim | LKSOM 2 | 5 Sep 2026 | 8.7452 |
| 5 | kparikh3 | CLEAR EC Direct Regression Seed 123 | 12 Sep 2026 | 8.7583 |
| 6 | mjkwon | GleeLAB | 23 Jul 2026 | 8.7720 |
| 7 | pbayat | cnn ensemble hmax | 15 Sep 2026 | 8.8185 |
| 8 | BascomPalmer | BP EC V3 | 14 Sep 2026 | 8.8660 |
| 9 | XueGao | ensemble1 | 15 Sep 2026 | 8.8956 |
| 10 | MxieR | mxier2 | 13 Sep 2026 | 8.9037 |

30 entries in total. Gap to 1st: 0.080 pp; 2nd–6th span 0.071 pp. Top-5 advance to Phase II (1,000 hidden images, 2 submissions).
Algorithm detail pages of other teams require sign-in (permission request); only two Penn-internal pages were public
(classical watershed + learned calibration, Phase I rank 25).

Implication for our Phase II estimates: our internal OOF deltas (hybrid −0.090 vs the Phase-1 package on fold 4) are of
the same order as the gap to 1st, but Phase I scores on 100 images have sampling noise of roughly ±0.1 pp, so Phase II
on 1,000 images will reorder teams. Phase II is the decisive one.

## Debugging Phase leaderboard (public, read 2026-10-30)
100 entries; the Phase I leader (abdulsohu) used it at least five times (V14/V15/V16, 1–23 Sep) before/around Phase I
submissions; LKSOM used it 5+ times. Scores there (3.1–3.8) are on the organisers' sample images and are not comparable
to Phase I/II. Latest visible debug submission: 23 Sep 2026. Whether the debug phase accepts submissions during Phase II
is only visible after sign-in (the submit page is login-gated). If it is open, it is the right place to verify the hybrid
container end-to-end on the platform (T4, time limit) WITHOUT spending one of the two Phase II slots.
