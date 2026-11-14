#!/bin/bash
# Lane GPU 3 runs lane GPU 1's end-to-end job for SSL arm cropopt_both, fold 0, seed 46, on GPU 3, as the coordinating session asked on 2026-11-07. Outputs stay in lane GPU 1's e2e folder.
# Guards: right GPU, finished encoder, not also queued on lane GPU 1, fresh root. Any failed guard exits non-zero, and the queue moves to its next job.
set -u
L1=/workspace/clear-ec-research/experiments/lane_gpu1_20261106
[ "${CUDA_VISIBLE_DEVICES:-}" = 3 ] || { echo "guard: CUDA_VISIBLE_DEVICES is '${CUDA_VISIBLE_DEVICES:-}', not 3"; exit 3; }
[ -f "$L1/ssl/cropopt_both_f0/run/complete.json" ] || { echo "guard: encoder not complete"; exit 4; }
if grep -qsE '^e2e_cropopt_both_f0[[:space:]]' "$L1/queue.tsv" "$L1/queue_short.tsv"; then echo "guard: lane GPU 1 still queues this job"; exit 5; fi
[ ! -e "$L1/e2e/cropopt_both_f0/f0_s46" ] || { echo "guard: root already exists"; exit 6; }
exec bash "$L1/source/run_e2e.sh" cropopt_both_f0 0 46
