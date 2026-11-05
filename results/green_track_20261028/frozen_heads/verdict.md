# frozen_heads — verdict (2026-10-31)

Head-only detectors on the frozen fold recipe A encoder (the hybrid's own s46 tokens at inference; zero extra
ViT-B passes, ~49 KB per head) keep about one third of the hex6 gain. Singles (pp HEX on A): stride-8 linear
probe -0.007; stride-2 heads s46 -0.022, s47 -0.018, s48 -0.033 (val-5 F1 0.57-0.62); srep head-only -0.051
(F1 0.56-0.58) — the external-only member is again the strongest of its family. Frozen set on base B:
-0.0335 pp HEX = -0.0112 pp mean, HEX 5/5 folds, 0/5 folds <= -0.03 (per fold -0.031 / -0.025 / -0.022 /
-0.018 / -0.013; the evaluator's default inclusion of the fine-tuned stride-8 s46 makes this slightly
flattering); active ten: -0.0311. Mixing the frozen heads into the fine-tuned set (15 detectors) dilutes it to
-0.0267. Trade-off for the PI/parent: -0.031 pp at 3.4 GB and ten extra passes, or -0.011 pp at ~zero cost.

Parent decision (2026-10-31): ACTIVE ten kept for submission 2 (T4 exec projected ~100-120 s + <= 30 s, image under ~8 GB);
the frozen family is the documented fallback only (deploy/fallback_frozen/), used if the T4 try-out fails on time and an A10G
does not fix it.
