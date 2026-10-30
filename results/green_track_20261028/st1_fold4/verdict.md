# st1_fold4 — verdict (2026-10-28)

Not extended. One round of confident-peak self-training (tau_pos 0.35 chosen by the pre-registered precision
rule on the 20 train-green images, tau_neg 0.10, 3000 mixed updates, 19 min) raises held-out point F1 on the
5 validation-green images from 0.642 to 0.666 (+0.024; recall 0.673 -> 0.714, precision 0.628 -> 0.641),
below the +0.03 bar. The pre-registered CD-correlation diagnostic failed on NaN handling; the NaN-safe
post-hoc check on the 1790 fold-4 validation rows shows no downstream change: hex6_all r 0.414 -> 0.423,
spacing-CD r 0.733 -> 0.733, median peaks 96 -> 108. Pseudo-label support is 87% confident negatives, so the
pseudo term mostly teaches suppression; a positive-balanced or hard-negative-mined round would be the next
variant if detector quality becomes the bottleneck for the HEX lead (the stride-2 head is tested first).
