# CLEAR-EC / CLEC 연구 인계 — 2026-10-13

## 0. 새 세션에서 가장 먼저 읽을 요약

- 사용자 요청: 다른 세션에서 이어서 작업할 수 있도록 현재 상태·목표·실험 이력·실행 방법을 상세 기록한다.
- **현재 진행 중인 작업은 SSL 5 epoch 기반 full tuning의 head 학습률 × gradient clipping sweep이다.** 새로운 4개 설정을 H200 2장으로 순차 학습 중이다. 기존 2개 결과를 재사용해 총 6개를 비교한다.
- **현재 Job을 다시 생성하거나 학습 스크립트를 재실행하지 말고 먼저 실제 상태를 조회한다.** 이 문서는 시점별 기록이며 실시간 상태가 아니다.
- 확인 시각: **2026-10-13 14:50:42 America/New_York (EDT), 18:50:42 UTC**.
- 해당 시각: Job active=1, Pod Running, train 컨테이너 ready=true, 재시작 0회. 첫 arm `head1e4_clip1`, **회귀 epoch 1/20, 2,800/7,161장, update 175**까지 로그 확인.
- 네 arm의 CPU 검사와 2GPU smoke 검사는 모두 통과했다. Encoder 전체가 학습되는 것도 확인했다.
- 자동 실행 범위: 네 arm의 각 20 epoch 학습 → 기존 두 결과와 검증 비교 → **최종 epoch20 평균 MAPE 기준 상위 2개 선정**.
- **다른 seed 재검증과 다른 fold의 SSL/회귀 재학습은 아직 구현·등록하지 않았다.** 자동 알림·주기 모니터도 설정하지 않았다.
- 예상 소요: 등록 시점부터 약 18시간. 현재 Job 시작은 2026-10-13 14:44:26 EDT. 초기 추정대로라면 10월 14일 오전 8:45 EDT 부근이나, 새 세션에서는 실제 epoch 속도로 재추정할 것.
- 본 문서를 작성하면서 학습 내용이나 실행 중인 작업은 변경하지 않았다.

## 1. 작업 위치와 접속 정보

| 항목 | 값 |
|---|---|
| 로컬 작업 루트 | `/Users/byounghanlee/Documents/kubectl` |
| 로컬 연구 저장소 | `/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research` |
| Git branch | `master` |
| 확인한 HEAD | `b6f77bf` — `Finalize verified Phase I artifacts and standalone training entry points` |
| Kubernetes namespace | `p-ajou-mlxp-project` |
| PVC | `clear-ec-workspace` |
| PVC mount | `/workspace` |
| 클러스터 연구 루트 | `/workspace/clear-ec-research` |
| Python | `/workspace/venv/bin/python` |
| 런타임 PYTHONPATH | `/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime` |
| 이미지 | `leebyounghan/mlx-rl:20260810-sgl0512-verl-1ddf071` |
| GPU 기본 정책 | H200 2장, 기존 프로토콜에는 여유 시 최대 4장 가능·다른 작업 선점 금지 명시 |

**Git 유의사항:** 최근 연구 코드·YAML·결과 디렉터리 상당수가 untracked 상태다. 이 세션에서는 commit/push하지 않았다. 문서 작성 전 `git status --porcelain`은 67개 항목이었다(파일 총수 아님). **GitHub clone만으로 최근 연구를 복원할 수 있다고 생각하지 말 것.** 로컬 파일과 PVC가 중요하다. `git clean`, reset, 폴더 일괄 삭제를 하지 말 것. 대형 모델·이미지는 로컬에 모두 복사되어 있지 않으며 PVC에 있다.

이 문서에서 아래의 `code/`, `results/`, `k8s/`, `docs/` 경로는 로컬 연구 저장소 기준이다. `experiments/` 경로는 원칙적으로 클러스터 연구 루트 기준이다. 동일한 이름의 로컬 `results/` 디렉터리는 일부 기록의 사본이지 전체 PVC 미러가 아니다.

## 2. 사용자의 목표와 최근 의사결정

### 장기 목표

기존 `results/phase2_foundation_20261002/protocol.json`에 기록된 목표:

- T4 16 GiB 배포 가능성을 전제로 단일 모델 정확도 최대화.
- 앙상블·증류보다 단일 모델 기반을 먼저 개선.
- 원본 전체 프레임 입력을 주요 방향으로 탐색.
- 과거 기록상 사용자 목표 완료일은 2026-11-05. 새 세션에서 일정 판단에 활용하되 새로 확정된 마감으로 주장하지 말 것.
- 후보: CPSAM v2, CellposeDINO ViT-B, DINOv3 ViT-B/16, ConvNeXt Base.
- 위 protocol의 `accuracy_training_started: false` 등은 **10월 2일 초기 상태**로 현재 상태가 아니다. 최신 상태는 본 문서와 현재 Job/결과를 따른다.

### 이번 대화에서 정해진 방향

1. LeJEPA SSL 5 epoch의 회귀 결과를 확인한 후 SSL을 총 10 epoch로 연장했다.
2. 10 epoch가 5 epoch보다 최종 평균 MAPE를 개선하지 못했다.
3. SSL 5 epoch 가중치를 고정하고 회귀 학습 파라미터를 점검했다.
4. Head LR 감소·encoder LR 감소·head-only 비교를 수행했다.
5. **기존 모델도 head-only가 아니라 encoder 전체를 학습하는 full tuning이었다.** 사용자에게 이를 명확히 설명했고, 별도의 head-only 대조군을 추가했다.
6. Head-only는 열세였고, head LR 감소는 좋은 중간 점수를 보였으나 마지막 점수는 개선하지 못했다.
7. Full tuning 두 실험에서 마지막 epoch의 모든 update가 clipping 기준 1.0을 넘는 것을 관찰했다. 이것을 버그로 단정하지 않고 clipping 기준도 sweep하기로 했다.
8. 사용자 승인: `오케이 진행해줘` — 아래의 추가 4개 sweep과 상위 후보 재검증 방향에 동의했다. 현재 실제 등록 범위는 **4개 sweep + 자동 비교/상위2 선정**까지이다.

공식 제출·배포·새로운 all-data fit·기존 제출물 교체는 이 세션에서 수행하지 않았다. 다음 세션은 실행된 범위와 계획만 있는 범위를 구분해 설명할 것.

## 3. 현재 실행 중인 sweep

### 리소스 및 실행 순서

- Job: `clear-ec-lejepa5-sweep-1013`
- 관측 Pod: `clear-ec-lejepa5-sweep-1013-9s487` (다음 세션은 label로 다시 조회)
- 관측 node: `h200-04-w-11f1`
- 컨테이너: `train`
- H200 2장, CPU request/limit 4/8, memory request/limit 20/48 GiB.
- 한 Job 안에서 **모든 arm smoke → 각 arm train 순차 실행 → audit**.
- 실패 시 `set -euo pipefail`로 이후 단계 중단. `backoffLimit: 0`, 자동 재시도 없음.
- Job deadline 172800초(48시간), Pod deadline 165600초(46시간).
- `ttlSecondsAfterFinished: 172800`: 종료 후 약 48시간 지나면 Job/Pod/log가 사라질 수 있음. **PVC 결과는 별도 유지된다.**
- 기존 CPU reader pod `clear-ec-lejepa-stage-1003`는 정리하여 삭제했다. 필요 시 아래 절차로 다시 생성할 수 있다.

### 조합

모든 조합은 SSL 5 epoch 초기 encoder, encoder LR `1e-5`, seed42, batch16, 회귀 20 epoch, full tuning이다.

| arm | Head LR | Gradient clipping | 처리 |
|---|---:|---:|---|
| `head1e3_clip1` | 1e-3 | 1.0 | 기존 10월 8일 full baseline 결과 재사용 |
| `head3e4_clip1` | 3e-4 | 1.0 | 기존 10월 12일 `full_head_low` 결과 재사용 |
| `head1e4_clip1` | 1e-4 | 1.0 | 새 실행 1, 마지막 확인 시 학습 중 |
| `head1e4_clip3` | 1e-4 | 3.0 | 새 실행 2, 앞 arm 완료 후 자동 시작 |
| `head3e4_clip3` | 3e-4 | 3.0 | 새 실행 3 |
| `head1e3_clip3` | 1e-3 | 3.0 | 새 실행 4 |

실제 learning rate는 1 epoch warmup 및 cosine schedule을 거치므로 초기 로그의 LR은 위 최대 LR보다 작다. 작은 LR 로그를 설정 오류로 오해하지 말 것.

### 코드·manifest·결과

- 로컬 runner: `code/archive/phase2_explore_2026-10/phase2_lejepa5_sweep_20261013.py`
- 로컬 Job YAML: `k8s/clear-ec-lejepa5-sweep-1013.yaml`
- 로컬 기록: `results/phase2_lejepa5_sweep_20261013/`
- PVC experiment: `experiments/phase2_lejepa5_sweep_20261013/`
- 실행용 frozen source: `experiments/phase2_lejepa5_sweep_20261013/source/phase2_lejepa5_sweep_20261013.py`
- 각 arm manifest: `experiments/phase2_lejepa5_sweep_20261013/arms/<arm>/manifest.json`
- 각 arm 검사: 같은 디렉터리의 `cpu_tests.json`, `smoke.json`
- 각 arm 학습 결과: 같은 디렉터리의 `runs/fold3/cpdino_vitb/`
- 전체 최종 비교: `experiments/phase2_lejepa5_sweep_20261013/audit.json` — 아직 생성 전일 수 있음.

로컬에 이미 저장한 기록:

- `dispatch.json`: 등록 정보, 자동화 범위.
- `arms/<arm>/manifest.json`, `cpu_tests.json`, `smoke.json`: 네 arm 모두 검증 사본 존재.
- `confirmation_preflight.json`: 다른 fold 평가에 필요한 사전 확인 결과.
- `handoff_snapshot_20261013.json`: 이관 직전 Job/Pod/최근 로그의 요약.
- `handoff_tail_20261013.log`: 이관 직전 로그 일부.

현재 runner SHA256:

```text
59565ccfe808abc7ea51b4b016ca529af09dd2529fab124fc26e2a15575f8d93
```

### 자동 비교/선정 규칙

- `audit()`가 기존 2개 + 새로운 4개의 completion/result/checkpoint/prediction hash, validation IDs·targets, epoch별 update 수·데이터 순서를 검사한다.
- 새로운 arm은 20개 epoch 가중치와 20개 prediction 사본이 있는지도 확인한다.
- 각 arm의 **최종 epoch20 mean MAPE**를 오름차순 정렬하고 동률은 arm 이름으로 정렬한다.
- `audit.json`의 `ranking`, `shortlist`가 결과이다.
- 중간 최저 epoch/점수 및 마지막 5개 epoch 점수 평균은 보조 진단이다. 이것은 weight averaging/ensemble 점수가 아니다.
- `confirmation_status`는 별도 seed/fold 확인이 아직 시작되지 않았음을 명시한다.
- **이 Job이 상위 후보를 다른 seed/fold로 자동 재학습하지는 않는다.**

## 4. 고정된 모델 및 학습 조건

### SSL 5 epoch

- CellposeDINO ViT-B encoder. 원래 외부 사전학습 가중치에서 시작한 LeJEPA adaptation.
- 원본 972×1296, 두 view, full-frame scale 보존. 일반적인 multi-crop LeJEPA 구현과 동일한 설정이라고 주장하지 말 것.
- SSL train 7,161개, validation 사용 0. 총 5 epoch, 1,120 optimizer updates.
- SSL batch: 이미지 32개 × 2 view, gradient caching micro-view 4.
- Encoder LR 1e-5, projector LR 1e-3, AdamW weight decay 1e-4.
- 첫 20 update warmup. 이후 5 epoch 연장 및 10 epoch 연장에서는 LR 유지, optimizer/RNG 복원. cosine decay를 사용한 회귀 단계와 구분.
- LeJEPA lambda 0.02, projector output128, SIGReg directions256, knots17.
- Augmentation: H/V flip, gain 0.9–1.1, offset −0.03–0.03. resize/crop/mask 없음.
- SSL5 가중치: `experiments/phase2_lejepa_total5_20261003/run/last.pt`
- SHA256: `6f1beb2d7dd1a34c7132cf65ca3cc80426c1428f6345b6569af1c538a7f39042`
- 회귀에는 SSL encoder만 로드하고 projector/SSL optimizer는 로드하지 않는다.

### 회귀 full tuning

- Encoder parameters: 85,669,632. Head parameters: 100,355. 총 85,769,987.
- Head: patch token mean pooling → LayerNorm → Linear128 → GELU → Linear3.
- 전체 native972×1296, bottom20/right16 replicate-padding → 992×1312.
- CLAHE + percentile(1,99) 0–1 normalization; CPDINO는 Cellpose01 입력.
- 회귀 augmentation은 H/V flip만 사용. 크기 변화 없음.
- 기존 full encoder/head LR = 1e-5 / 1e-3. 현재 sweep은 head LR 및 clipping만 변경.
- AdamW weight decay1e-4, BF16, warmup1 epoch, 이후 cosine → 최대 LR의 10%.
- Seed42. 각 epoch 순서는 NumPy seed `42+zero_based_epoch`.
- Global batch16; 2GPU DDP, rank micro-batch4, rank accumulation2.
- 마지막 9개 샘플도 포함. 빈 rank microbatch는 zero-weight dummy forward로 동기화.
- 학습 샘플7,161, validation1,791, 회귀20 epoch, epoch당448 updates, 총8,960.
- 손실: `0.5 * standardized log L1 + 2.5 * abs(expm1(log prediction error))`.
- 지표: CD/CV/HEX별 MAPE(%)의 산술평균. **낮을수록 좋다.** 0.05 차이는 MAPE 점수 0.05 percentage point이며 5% 향상이 아니다.
- 평가: 매 epoch 전체 이미지 1 view, TTA/보정 없음, 원래 unsmoothed labels, train-only log mean/std.
- Head 초기화·validation rows·loss·데이터 순서를 맞춘 비교다. GPU/확률적 연산으로 bitwise 동일성을 보장하지 않는다.

### 검증 내용과 기록 개선

- CPU: metric/loss identity, zero-target rejection, partial batch 가중치, telemetry 샘플 가중치, clipping 기준1/3 실제 gradient scale 검사.
- GPU: single-GPU reference 대비 DDP gradient relative L2 <0.002; 9개 샘플·empty microbatch 포함; rank 동기화; 12개 encoder block에 gradient 도달; encoder/head 실제 변경 확인.
- 현재 네 arm 모두 full tuning이다. 코드에 남아 있는 head-only 분기는 이전 runner 재사용 흔적이며 현재 ARMS에는 head-only가 없다.
- Epoch별 validation predictions와 model-only 가중치를 저장한다.
- Online train loss/MAPE, standardized log L1, gradient norm 평균·최대, clipping 비율, LR을 기록한다.
- 별도의 고정된 training probe256개(seed12012)를 eval mode/no augmentation으로 평가한다.
- Probe 평가가 다음 epoch 난수 상태를 바꾸지 않도록 `torch.random.fork_rng`로 보호했다.
- Online train MAPE는 augmentation된 샘플과 매 update 변경되는 가중치로 측정한다. 고정 checkpoint 평가와 같다고 해석하지 말 것. Train probe 역시 전체 train이 아닌256개이다.

## 5. 완료된 실험 결과

### SSL 기간 비교: 모두 회귀20 epoch 최종 점수

| SSL | CD | CV | HEX | 평균 MAPE |
|---|---:|---:|---:|---:|
| 추가 SSL0 | 7.2997 | 10.2319 | 10.3736 | 9.3017 |
| LeJEPA1 | 7.1400 | 10.2716 | 10.3328 | 9.2481 |
| LeJEPA5 | 7.0587 | 10.3406 | 10.3215 | **9.2403** |
| LeJEPA10 | 7.0849 | 10.3863 | 10.2624 | 9.2445 |

- 5→10의 평균 변화 +0.004261: 더 좋아지지 않았다. HEX는 개선, CD/CV 악화.
- SSL5 이후 추가 SSL5 epochs에4.75h, 후속 회귀4.27h 소요.
- 기존 SSL0/1은 1GPU, SSL5/10 회귀는2GPU. Global batch/update는 일치하지만 실행 방식 차이가 있다.
- 결과: `results/phase2_lejepa10_regression_20261009/audit.json`

### SSL5 회귀 튜닝 / full versus head-only: 모두 최종 epoch20

| 설정 | Encoder LR | Head LR | CD | CV | HEX | 평균 | 기존 대비 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 full baseline | 1e-5 | 1e-3 | 7.0587 | 10.3406 | 10.3215 | **9.2403** | — |
| `full_head_low` | 1e-5 | 3e-4 | 7.2219 | 10.3486 | 10.3052 | 9.2919 | +0.0516 |
| `full_encoder_low` | 3e-6 | 1e-3 | 7.3711 | 10.2961 | 10.1685 | 9.2786 | +0.0383 |
| `head_only` | 고정/0 | 1e-3 | 8.4789 | 11.1250 | 10.7134 | 10.1058 | +0.8655 |

- 세 신규 실험 완료·최종 audit 통과. 합계 약10h40m. Head-only는 약1.87h, full 두 개는 각각 약4.35h.
- Head-only는 encoder0개, LayerNorm/MLP head100,355개만 학습하는 것을 GPU smoke와 epoch마다 확인했다.
- Head-only가 열세인 것은 이 조건에서의 실측이며 모든 head-only 방식의 보편적 결론은 아니다.
- 두 full 실험 모두 마지막 epoch gradient clipping 비율1.0(100%). Clipping이 잘못됐다는 확정 근거는 아니며 다음 sweep의 동기다.
- 결과: `results/phase2_lejepa5_tuning_20261012/audit.json`
- 전체 로그: 같은 폴더 `completed_training.log`.
- Epoch 기록: 같은 폴더 `arms/<arm>/logged_history.json`.

### 중간 최저점: 사후 진단이며 최종 성능과 혼용 금지

| 설정 | 최저 회귀 epoch | 당시 평균 MAPE | 가중치 보존 |
|---|---:|---:|---|
| 기존 SSL5 full baseline | 14 | 9.1367 | **없음** — 당시 last.pt만 덮어씀 |
| `full_head_low` | 7 | 9.1049 | 있음, epoch_07.pt |
| `full_encoder_low` | 18 | 9.1269 | 있음, epoch_18.pt |
| `head_only` | 18 | 10.0445 | 있음, epoch_18.pt |

- 기존 full baseline은 7 epoch9.1374, 14 epoch9.1367, 마지막20 epoch9.2403이었다. 무조건 더 길게 학습하면 좋아진다는 근거는 없다.
- 현재 최고 중간 점수9.1049를 독립 성능으로 홍보하거나 최종 MAPE9.2403과 같은 선택 기준인 것처럼 비교하지 말 것.
- 기존 baseline에는 train loss가 없어 과적합을 단정할 수 없었다. 새 로그의 train/validation curves를 분석할 것.
- 기존 SSL5 최종 예측은 낮은 target을 과대, 높은 target을 과소 예측하는 경향이 있다. CV 하위 decile signed relative bias+17.0%, 상위−14.4%. `results/phase2_lejepa5_regression_20261008/parameter_diagnostics_20261012.json` 참조.

## 6. 실험 디렉터리 지도

아래 경로는 PVC 연구 루트의 `experiments/` 아래이다. 로컬에는 동명 `results/` 사본 일부가 있다.

| 실험 | 디렉터리 | 중요한 파일 |
|---|---|---|
| SSL0 회귀 | `phase2_foundation_train_20261002` | `runs/fold3/cpdino_vitb/{result.json,last.pt,predictions.npz}` |
| SSL1 자체 학습 | `phase2_lejepa_20261003` | `pilot/last.pt`, `pilot/result.json` |
| SSL1 회귀 | `phase2_lejepa_regression_20261003` | `runs/fold3/cpdino_vitb/`, `audit.json` |
| SSL5 자체 학습 | `phase2_lejepa_total5_20261003` | `run/last.pt`, `run/result.json`, frozen manifest |
| SSL5 기존 full 회귀 | `phase2_lejepa5_regression_20261008` | `runs/fold3/cpdino_vitb/`, `audit.json` |
| SSL10 자체 학습 | `phase2_lejepa_total10_20261009` | `run/last.pt`, `run/result.json` |
| SSL10 회귀 | `phase2_lejepa10_regression_20261009` | `runs/fold3/cpdino_vitb/`, `audit.json` |
| LR/head-only 비교 | `phase2_lejepa5_tuning_20261012` | `arms/<arm>/runs/fold3/cpdino_vitb/`, `audit.json` |
| 현재 sweep | `phase2_lejepa5_sweep_20261013` | `arms/<arm>/runs/fold3/cpdino_vitb/`, 추후 `audit.json` |

10월12일 이후 arm의 `runs/fold3/cpdino_vitb/` 파일:

```text
progress.json                    현재 진행 상황
history.json                     완료된 epoch별 점수와 진단
last.pt                          가장 최근 model + optimizer, epoch마다 교체
epoch_01.pt ... epoch_20.pt       model-only epoch 사본
predictions_epoch_01.npz ...      epoch별 validation 예측
predictions.npz                  최종 validation 예측
result.json                      최종 점수, history, hashes
complete.json                    정상 완료 표시 + result SHA256
```

`last.pt`는 rank1 RNG를 저장하지 않으며 **정확한 resume를 보장하지 않는다.** Epoch 사본은 optimizer가 없는 model-only 파일이다. 중단 시 단순히 해당 checkpoint를 불러서 같은 실험을 이어 돌렸다고 주장하지 말 것.

## 7. 새 세션에서 확인하는 명령

### 실행 상태 조회 — 가장 먼저

```bash
cd '/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research'
kubectl --request-timeout=20s -n p-ajou-mlxp-project get job clear-ec-lejepa5-sweep-1013 -o wide
kubectl --request-timeout=20s -n p-ajou-mlxp-project get pods -l job-name=clear-ec-lejepa5-sweep-1013 -o wide
kubectl --request-timeout=20s -n p-ajou-mlxp-project logs job/clear-ec-lejepa5-sweep-1013 -c train --tail=8
```

Pod가 Running이면 출력된 Pod 이름으로 `kubectl exec ... -c train -- ...`를 이용해 PVC JSON을 읽을 수 있다. 완료된 컨테이너에는 exec/cp가 되지 않으므로 아래 CPU reader를 사용한다. Job이 없다고 실패라고 단정하지 말 것(TTL 정리 가능).

### 로그 보존

```bash
mkdir -p results/phase2_lejepa5_sweep_20261013
kubectl --request-timeout=30s -n p-ajou-mlxp-project logs job/clear-ec-lejepa5-sweep-1013 -c train > results/phase2_lejepa5_sweep_20261013/completed_training.log
```

이 명령은 현재 로그를 덤프한다. 파일 이름이 `completed_training.log`라는 이유만으로 완료 상태가 확인되는 것은 아니다. Job status와 `audit.json`/`complete.json`을 함께 확인할 것.

### 종료 후 PVC 결과 읽기 — CPU reader

아래 manifest는 CPU-only 요청이다(GPU node에 스케줄되지만 GPU resource request 없음). 존재 여부를 먼저 확인하고 필요할 때만 생성한다.

```bash
kubectl --request-timeout=20s -n p-ajou-mlxp-project get pod clear-ec-lejepa-stage-1003
# NotFound일 때:
kubectl --request-timeout=30s -n p-ajou-mlxp-project apply -f k8s/clear-ec-lejepa-stage-1003.yaml
kubectl --request-timeout=20s -n p-ajou-mlxp-project wait --for=condition=Ready pod/clear-ec-lejepa-stage-1003 --timeout=45s
kubectl --request-timeout=30s -n p-ajou-mlxp-project exec clear-ec-lejepa-stage-1003 -c stage -- cat /workspace/clear-ec-research/experiments/phase2_lejepa5_sweep_20261013/audit.json
kubectl --request-timeout=30s -n p-ajou-mlxp-project cp clear-ec-lejepa-stage-1003:/workspace/clear-ec-research/experiments/phase2_lejepa5_sweep_20261013/audit.json results/phase2_lejepa5_sweep_20261013/audit.json -c stage
```

- `audit.json`가 없으면 각 arm의 `progress.json`, `history.json`, `complete.json`과 로그를 조사한다.
- 결과 JSON만 먼저 복사한다. `arms/` 전체를 무심코 복사하면 수십 GB의 가중치도 전송된다.
- 각 arm의 manifest/result/complete/history 및 필요한 predictions를 가져오고 hashes를 확인한다.
- 필요 작업 종료 후 **자신이 만든 reader만** 정리한다:

```bash
kubectl --request-timeout=20s -n p-ajou-mlxp-project delete pod clear-ec-lejepa-stage-1003 --wait=false
```

### 완료 검증 기준

1. Job `Complete` 또는 각 arm `complete.json` status=`completed`.
2. `complete.json.result_sha`와 `result.json` 실제 SHA256 일치.
3. Result의 manifest/prediction/checkpoint SHA256 일치.
4. 네 arm 각각20 epochs, 8960 updates, epoch snapshot20개.
5. 전체 `audit.json.status == ssl5_lr_clip_sweep_verified`.
6. 6개 설정 점수·delta·ranking·shortlist를 읽고 사용자에게 완료와 아직 남은 재검증을 구분해서 보고.

## 8. 실패/재시작 시 주의

- Frozen source/manifest는 파일 hash로 검사한다. **실행 중인 PVC source를 수정하지 말 것.** 로컬 source 변경도 현재 Job에는 자동 반영되지 않는다.
- `prepare()`는 기존 manifest가 있으면 거부한다. `smoke()`는 기존 smoke 파일이 있으면 거부한다. `train()`은 기존 run 디렉터리가 있으면 거부한다.
- 따라서 동일 YAML/runner의 단순 재실행은 복구 방법이 아니다. 앞서 통과한 smoke 때문에 후속 train에 도달하지 못할 수도 있다.
- 순차 Job이 중간에 실패하면 이미 완료된 arm과 checkpoints를 보존하고, 실패 원인 및 다음 미완료 단계부터 분리한 새 Job/새 실험 ID를 설계한다.
- 실패 이유가 일시적인 연결 시간 초과인지 실제 학습 실패인지 구분한다. 조회가 실패했다고 Job을 다시 등록하지 말 것.
- 체크포인트 optimizer/RNG 제약 때문에 재개를 구현할 경우 새 실험임을 명시하고 검증한다.
- 기존 로그의 IPv6 `Address family not supported` 경고, DDP `find_unused_parameters=True` 경고는 이 실행에서 smoke 통과·학습 진행과 함께 관찰됐다. 경고만으로 실패로 판단하지 말 것.
- 인증정보나 token은 이 문서에 저장하지 않았다. kubeconfig/토큰 내용을 보고서에 출력하지 않는다.

## 9. 후속 재검증의 실제 제약

### 다음 할 일

1. 현재 sweep이 완료되면 audit와 전체 학습 곡선을 수집한다.
2. 사전 선언한 terminal20 기준 shortlist2를 확인한다. 중간 최저점만 보고 선정 규칙을 사후 변경하지 않는다.
3. 동일 fold에서 다른 seed로 두 설정의 안정성을 확인하는 별도 실험을 준비한다. 아직 등록되지 않았다.
4. 다른 fold 검증은 그 fold의 training split만 사용해 SSL부터 다시 학습한다.
5. 후보가 재현되면 더 빠른 LR decay, loss 기여도/형태 정보, 최종 T4 추론 검증 등을 검토한다.

### Seed 변경은 recipe 숫자만 바꾸면 안 됨

현재 기반 코드에 seed42가 여러 곳에 고정되어 있다:

- `phase2_foundation_train_20261002.py`의 `seed()`.
- `setup()`의 head 초기화 `torch.manual_seed(42)`.
- `Frames.__getitem__()` augmentation SeedSequence `[42, epoch, idx]`.
- sweep train의 `default_rng(42+epoch)`.
- 기존 manifest의 `expected_head_sha`는 seed42 초기 head에 대한 값.

다른 seed 확인 시 이 경로를 일관되게 parameterize하고 새 초기 head hash·data order·manifest를 생성해야 한다. Manifest `seed`만 바꾸고 실제 seed42가 유지되면 독립 seed 실험이 아니다. SSL seed를 고정한 회귀 seed 재현성인지 SSL부터 바꾸는 재현성인지 구분해 기록할 것.

### 다른 fold와 정답0 문제

- 현재 준비된 `experiments/overnight_target_20260912/preflight/` split은 fold2와 fold3를 확인했다.
- Fold3: train7161/val1791, 세 target 모두 양수.
- Fold2: train7153/val1791, **validation HEX0이 2개**. Train은 양수.
- 현재 `b.arrays()`/`metrics()`는 target>0을 요구하여 fold2에 그대로 적용할 수 없다. Zero를 조용히 삭제하거나 epsilon으로 바꾸지 않는다. 공식 평가 규칙을 확인하고 별도 프로토콜로 명시해야 한다.
- 확인된 과거 audit에서 두 fold 모두 slide overlap0, exact-pixel overlap0, slide/exact 연결 group overlap0이었다.
- **Patient independence와 near-duplicate 검증은 아직 미해결**. Slide가 다르다고 patient 독립성을 주장하지 않는다.
- 현재 SSL5는 fold3 training split으로 학습했다. 다른 fold의 validation 이미지 상당수가 이 SSL training에 포함됐을 수 있으므로 **현재 SSL5/9.1049 checkpoint를 다른 fold에 그대로 평가해서 독립 검증이라고 할 수 없다.**
- Fold2 역시 과거 연구에서 사용한 development fold이다. 다른 fold 재학습은 재현성 근거를 늘리지만 자동으로 완전한 untouched holdout이 되는 것은 아니다.
- 근거: `results/phase2_lejepa5_sweep_20261013/confirmation_preflight.json`.

### 배포/공식 점수

- 이번 값들은 재사용된 내부 fold3의 단일 모델 MAPE이다. 이전 제출/앙상블 점수8.7010이나 다른 sample/filter/TTA 조건의 점수와 직접 비교하지 말 것.
- H200 BF16에서 학습 성공은 T4 16 GiB 추론 가능성의 증명이 아니다. T4 검증은 남아 있다.
- 상위 모델의 공식 제출, 기존 Docker/model artifact 변경은 아직 별도 단계다.

## 10. 다른 후보와 우선순위

- 현재 우선: SSL5 full tuning LR/clipping sweep → 재현성 확인.
- SSL20 epoch: 제안만 있었고 실행하지 않았다. SSL10이 최종 평균에서 개선하지 못해 우선순위 낮아졌다.
- 일반 DINOv3 ViT-B/16: 가중치 다운로드·CPU load 확인 기록 있음. CellposeDINO와 같은 조건으로 비교하는 후보.
- CPSAM native512: 과거1 epoch warm-start screen에서256 대비 평균0.2454 개선. 다른 fold/긴 학습 확인 필요. 현재 점수와 직접 비교 불가.
- 세포 형태/유효 영역 branch: tail error와 평균 회귀 문제를 겨냥한 후보.
- 과거 CPSAM masked-EMA SSL screen은 screening gate 실패; near-identical seed 추가는 우선순위 낮음.
- ConvNeXt는 과거 중앙 영역 비교에서 CPSAM보다 낮은 성능; 현재 우선 후보 아님.
- Calibration/ensemble/distillation은 단일 모델 개선·독립 검증 후 고려.
- 관련 근거 문서: `docs/PHASE2_ROADMAP_20260928.md`, `results/phase2_cpsam_scale_20261001/README.md`, `results/phase2_backbone_screen_20261001/audit.json`, `results/ensemble_ablation_20260923/README.md`.

## 11. 새 세션의 첫 응답에 필요한 체크리스트

- [ ] 이 문서의 기록 시각과 현재 시각을 구분한다.
- [ ] Job 상태·현재 arm/epoch를 실제 조회한다.
- [ ] 진행 중이면 중복 실행하지 않는다.
- [ ] 완료됐으면 audit·학습 곡선·snapshot 존재를 확인하고 로컬로 결과 기록을 가져온다.
- [ ] 기존 최종 기준9.240254402160645와6개 설정을 동일 기준으로 비교한다.
- [ ] 최종/중간 최저점/평균화된 진단 통계를 구분한다.
- [ ] 상위2의 다른 seed 확인은 새 코드·Job이 필요하다는 점을 인지한다.
- [ ] 다른 fold 확인은 fold별 SSL 재학습과 zero-target 정책 확인이 필요하다.
- [ ] 어떤 후속이 실제 실행 중인지, 제안/미등록인지 명확히 알린다.

## 12. 이관 시점의 범위

이 문서는 상태를 남기는 작업이다. 새 채팅 생성·다른 채팅으로 메시지 전송·자동화 등록은 수행하지 않았다. 현재 Kubernetes Job은 이 채팅 종료와 별개로 실행된다. 문서와 로컬 연구 폴더 및 클러스터 접근이 있는 새 세션에서 위 절차로 이어받으면 된다.
