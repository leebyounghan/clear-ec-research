---
id: method-t4-fp16-one-detector-copy
type: method
name: 'T4 dtype path: fp16 autocast with one detector copy'
summary: 'Container default below CUDA compute capability 8.0 (a T4 is sm75, no bf16 kernels): every bf16 autocast region runs in fp16, Cellpose in fp32, and the 13 non-pre-only detectors run one copy of their single view instead of a batch of four identical copies (the copy count changes outputs by < 5e-7 relative).'
code: lane_cellroute_20261109/inference/pack_hooks.py; inference/test_t4.py; inference/compare_t4.py; inference/impact_t4.py
---

# T4 dtype path: fp16 autocast with one detector copy

Container default below CUDA compute capability 8.0 (a T4 is sm75, no bf16 kernels): every bf16 autocast region runs in fp16, Cellpose in fp32, and the 13 non-pre-only detectors run one copy of their single view instead of a batch of four identical copies (the copy count changes outputs by < 5e-7 relative).

## Where it is used

Experiments link here with `uses_method`; see backlinks. Sources: ledger O-DEPLOY-016, A-DEPLOY-009, O-DEPLOY-020; `lane_cellroute_20261109/inference/pack_hooks.py` (CLEAR_EC_AUTOCAST, CLEAR_EC_DET_COPIES, CLEAR_EC_PREC).
