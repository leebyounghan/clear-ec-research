# September 5 — continue performance exploration before September 10

Latest status: [September 6 seed replication and confirmation](REGION_SEEDS_20260906.md).
The launch/status descriptions below are historical; both original queues completed.

## Authorization and protected work

User explicitly requested continued crop/representation/annotation exploration
before freezing the September10 candidate. This wave uses the existing H200 pod;
it does not create/delete/stop pods, upload/submit, or overwrite prior models.
The corrected all-data candidate and postflight remain on GPU0 under
`experiments/deploy_review_20260905`. Its finished OOF numerical checks are not
an independent performance test. Do not interrupt that queue.

New root: `/workspace/clear-ec-research/experiments/region_search_20260905`.
Frozen source lives in `source/code`; manifest records its hashes. The original
`code/train_method_h200.py` is preserved; new provider hooks are isolated in
`train_region_method.py`. Existing regression architecture, slide folds,
training-only slide smoothing and optimizer logic are unchanged.

## Hypotheses, not another repeat of the old full-frame method

Earlier two-fold global/local feature-attention results were worse than their
central native control:9.416639 full/uniform,9.400113 full/focus vs9.152901 central.
That setup mixed a resized256 global image with a native tile. The new three
policies use ONLY native256 crops from the972×1296 original, prediction-level
averaging, and the newer mixed-MAPE/cosine recipe. No resized global view enters
the network. All supervised regional candidates have identical batch48, two
training crops,35epochs (149 updates/epoch), and5×D8=40 evaluation views.

| Recipe | Training distribution | Five evaluation locations |
|---|---|---|
| full_uniform | Two independent uniform native crops anywhere in the original | Full-field corners + centre |
| full_quality | Two crops from eight high-focus, spatially separated proposals; jitter±32pixels | Top five diverse proposals |
| anchor_quality | Each crop has60% central-window /40% full-quality sampling | Three central anchors + two diverse quality proposals |

Quality ranking uses the existing unlabeled contrast/Laplacian/saturation
heuristic and deterministic spatial separation, not ROI ground truth. It can
still select reflections or nonrepresentative fields; the uniform/anchored
controls and downstream folds determine usefulness. No crop policy is selected
from training loss. This is an ablation of image sampling, not proof of ROI recovery.

Native cache contains8,954 images in exactly the original OOF order. All9,000
originals remain available separately for later whole-data retraining; do not
change the evaluation population mid-comparison.

## Additional external-only pretraining

Prior40epoch small-source JEPA and55epoch all-source/FECD JEPA both completed;
the small-source optimizer-corrected OOF9.225720 was not better than the
corrected original CPSAM9.209081. Merely repeating that setup is not the plan.

New pretraining starts from original CPSAM and uses378 verified SREP specular
images plus52 RODREP endothelium images, with85%/15% source sampling. No CLEAR-EC
images or labels are read by this additional pretrainer, so its subsequent
fold evaluation is not transductive through CLEAR-EC image pretraining.
Original pretrained CPSAM's upstream corpus is not newly audited here.

Configuration:6,000 updates, batch32 (192,000 sampled image views), LR1e-5 with
200-update warmup/cosine floor, BF16,40–60% block masking, EMA teacher and a
spatial3×3 predictor. Native cropping/reflect padding replaces scale jitter;
photometric changes are mild and no Gaussian blur is added. This is a custom
masked EMA feature-prediction experiment inspired by JEPA, NOT a reproduction
of official I-JEPA. Record feature variance and finite losses to catch gross
collapse; SSL loss alone does not establish downstream improvement.

Each external file is hashed; config stores corpus counts, source weights,
image hashes and manifest hash. Checkpoints are separate every1,000 updates,
including online/teacher/optimizer/RNG state. Restart is not automatic.
Final encoder's hash and `external_only` config must match before downstream
training can initialize from it; all-data supervised weights are forbidden.

After pretraining, `native_ssl` runs central single-crop metric35 on folds0/1,
matching that supervised recipe. The same external-only encoder can initialize
later confirmation folds without seeing CLEAR-EC labels/images during pretrain.

Primary method reference: [official I-JEPA repository](https://github.com/facebookresearch/ijepa)
and [original paper](https://arxiv.org/abs/2301.08243). Native crop choices and
the spatial predictor are our experimental modifications, not claims from those sources.

## Fixed selection and finite automatic progression

First screen: three crop recipes × folds0/1, plus the external-pretrained recipe
× folds0/1 after pretraining: eight supervised jobs in total.

For every candidate compare exactly aligned predictions with the CURRENT
five-model baseline, not an obsolete weaker model. No fitted member weights:

- add6 =5/6 current5 +1/6 new model;
- replace4 =3/4 old three-seed mean +1/4 new model.

Advance a recipe only if add6 raw overall improves at least0.01 points AND both
screening folds improve, OR replace4 improves at least0.005 points AND both
folds improve. Each passing recipe then runs folds2/3/4, reports those separately
and reports all five folds. At most20 supervised fits + one pretrain are in this
finite wave (12 confirmation fits if all four recipes pass). Every job first
passes a one-update/two-image GPU smoke. Full job limit12h; smoke15min. No passing
flag automatically replaces/deploys a candidate; target-runtime validation remains.

These folds have been repeatedly used historically and checkpoint selection
still uses validation. Cross-fit calibration is exploratory, not fully nested.
Confirmation folds are unused for THIS two-fold decision, not a pristine test set.

## Preflight and actual launch

Five new tests plus seven existing method tests passed (12 total): native pixel
preservation/no resize, deterministic distinct crop origins, bounds/D8/view
counts, unchanged network/folds, external-only source selection, frozen training
budget and rejection of candidates that only beat an obsolete baseline.

Diagnostic proposals on four fold0 TRAINING rows were visually inspected.
Peripheral cell patterns remain visible; large black/reflection regions occur
in some originals. This supports testing sampling policies but is not a scoring
result. The plot `results/native_region_proposals_20260905.png` contains
algorithm-proposed crop boxes, NOT manually validated ROI/cell annotations. No
new CD/CV/HEX labels, cell masks or pseudo-GT were inserted into training.

Controller launched around20:55 UTC September5, PID67473. All seven GPU smokes
passed, then all seven full jobs started. The scheduler reassigns free GPUs after
smokes, so actual20:56UTC placement differs from the initial smoke order:

| GPU | Actual full job | PID |
|---|---|---:|
| 0 | Existing corrected deployment fit (separate controller) | 62217 |
| 1 | External-only native SSL | 68256 |
| 2 | anchor_quality fold1 | 68264 |
| 3 | anchor_quality fold0 | 68269 |
| 4 | full_quality fold1 | 68275 |
| 5 | full_quality fold0 | 68278 |
| 6 | full_uniform fold1 | 68284 |
| 7 | full_uniform fold0 | 68290 |

Eight GPUs have active jobs. Utilization fluctuates during CPU preparation and
validation; do not claim100% duty cycle. Regional allocation is about50.56GiB
(51,769MiB in `nvidia-smi`); external SSL was about21GiB
at its first snapshot. No new node or external data upload was needed.

## Timeline and continuation

### Actual checkpoint: September 5, 23:19 UTC

External-only pretraining completed22:55:58 UTC,6,000 updates in7,218.63s.
Final encoder SHA256: `4d5eb894e85dbc685077d0528f62e753a7ac83ae5ab3e792b2c21f2d3a5d25f1`.
GPU1 automatically started native_ssl fold0 (PID78190); at23:19 it had
finished4/35epochs, about290–292s/epoch. Fold1 is pending a free GPU1–7.
The six crop fits remain active without failures. GPU0's separate deployment
postflight finished22:51; GPU0 was idle at23:19 (28minutes), seven GPUs active.
Do not manually run the pending fold into the existing scheduler-owned output
path, or modify the running frozen controller to reclaim GPU0.

Final SSL target-token standard deviation was0.0831 versus roughly0.5–0.7
early in training. This decline warrants representation-collapse investigation;
neither the finite-loss gate nor the low SSL loss proves useful representation.
No candidate promotion is justified before downstream matched fold results.

Historical matched two-crop training took about526s/epoch:35×526/3600≈5.11h
before final D8 inference/startup. Initial regional screen should be available
around September6 02:15–02:45UTC if throughput transfers. SSL timing is measured
from actual progress, not inferred from its tiny smoke: at20:57UTC it completed
100updates in120.55s (~1.206s/update), projecting about2.0h plus checkpoint I/O,
roughly23:00–23:15UTC for pretraining. Target feature std was0.6679 and loss was
finite; this is an anti-collapse diagnostic, not a downstream performance score.
Its downstream
single-crop35epoch fits previously took roughly2.8h each; confirmation adds
roughly3–5.5h per parallel wave. Conditional maximum compute is approximately
15 regional fits×5.1h +5 single-crop fits×2.8h +SSL, about90GPU-hours plus SSL
and inference overhead. This is compute planning, not billed node-hours or cost.

Use September6 results to decide the next bounded experiment: seed replication,
error-guided quality/region refinement, or another representation setting with
a concrete failure hypothesis. Do not spend until September10 blindly tuning
the same validation score. Aim to finish long exploration by September8 and use
September9 for chosen all-data fits, inference and container/target checks before
the requested September10 submission. September14 remains the second submission
target. Submission itself still requires explicit confirmation.

The existing30-minute `clear-ec` monitor now covers BOTH this search and the
protected deployment queue, using OpenAI Docs scheduled-task guidance. It should
not delete itself when only the older queue completes. It can continue bounded,
in-scope follow-up experiments under the user's explicit request, preserves
active jobs, and reports only meaningful changes/failures/user actions. Local
checks can arrive late; the pod controller advances this finite wave independently.
Docker builder storage/target-driver questions remain separate from GPU research;
no user files, images or Docker caches may be deleted to free space implicitly.
