# CPSAM native input-field screen (2026-10-01)

This is a bounded Phase II research experiment, not a submitted model.

- Arms: centered native 256×256, central native 512×512, and uncropped native 972×1296 (four edge-padded rows to 976×1296). No image is resized.
- All arms start from the same fold-2, 35-epoch central256 CPSAM control checkpoint. They train for one additional epoch on the same 7,153 fold-2 training IDs, in the same order, with batch 1 and the same optimizer, labels, augmentations, and fixed terminal checkpoint policy.
- Encoder positional embeddings are interpolated for the two larger inputs. CLAHE grid spacing is kept near 32 native pixels in each arm. There is no ensemble, learned view aggregation, validation-based checkpoint selection, or official-release change.
- Validation uses one deterministic centered/full view for the same 1,791 fold-2 IDs. The primary duplicate-clean subset has 1,781 rows; zero labels are omitted only from the corresponding MAPE denominator.
- Outputs and checkpoints reside on the shared PVC at `experiments/phase2_cpsam_scale_20261001/`. The local `audit.json` was copied here after all three arms finished and their IDs, training order, and hashes passed the audit.
- **Interpretation:** This one-epoch warm-start screen tests feasibility and directional effects of field of view. It is not directly comparable with the earlier 35-epoch/40-view CPSAM 9.1338 MAPE, nor is it an independent Phase II score. A favorable result needs confirmation on another fold and with a longer learning curve.

Kubernetes Job: `p-ajou-mlxp-project/clear-ec-p2-cpsam-scale-1001` (three indexed runs, at most two H200 GPUs concurrently, zero automatic retries). The audit is a separate CPU-only Job and runs only when all three arms are complete.

## Audited fold-2 result

Mean MAPE across CD/CV/HEX on the 1,781 duplicate-clean validation rows; lower is better:

| Input | Overall | CD | CV | HEX | Train time | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Center 256×256 | 9.6909 | 6.9697 | 11.5312 | 10.5717 | 14.5 min | 5.74 GiB |
| Central 512×512 | **9.4455** | **6.6761** | **11.2725** | **10.3879** | 44.1 min | 8.69 GiB |
| Native full 972×1296 | 10.3498 | 8.2682 | 11.9398 | 10.8414 | 14.8 h | 98.86 GiB |

The 512 arm improves on 256 by 0.2454 MAPE. The full-native arm worsens by 0.9043 versus 512 at this one-epoch budget. This does **not** establish that full-native input is intrinsically inferior: its larger position grid, global mean pooling, and limited adaptation are plausible confounders. The next gate is fold-3 replication and a longer 512 learning curve before promoting a foundation model.
