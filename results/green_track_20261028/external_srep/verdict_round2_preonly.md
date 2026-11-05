# external_srep — round 2, pre-trained-only member (2026-10-30 18:50 UTC)

GATE PASS on base B (PI rule). The stride-2 head pre-trained on srep (2000 updates, nominal 1.30x scale, per fold
from the fold's recipe A encoder) and used WITHOUT the 20-image fine-tune: val-5 F1 0.54-0.59 (the lowest of all
detectors) yet the strongest single hex6 member (-0.064 pp HEX on A). Added as a tenth member to the frozen nine:
- base B: -0.0934 pp HEX = -0.0311 pp mean, 5/5 folds improving, 3/5 folds <= -0.03
  (per-fold mean -0.0473 / -0.0309 / -0.0415 / -0.0274 / -0.0086); the nine alone: -0.0286.
- base A: -0.154 pp HEX = -0.0515 pp mean, 4/5 folds.
Mechanism: independence — every other detector was fine-tuned on the same 20 images; this one never saw them,
so its hex6 errors are less shared. Caveats: marginal on the mean; one of five pre-registered round-2 variants;
does not clear the parent's freeze rule for the deployed nine (+0.0025 pp, lower F1). Confirmation queued: two
more pre-only seeds (47, 48) and a 12-detector blend; recommendation threshold <= -0.035 on B with >= 3 folds.
Reported to the parent 18:50 UTC.
