# 단일 모델 SSL representation 실험 — 2026-09-13

## 완료 결과 (21:48:49 UTC)

네 실행과 최종 감사 성공, 실패 없음. 동일 3,572 primary 이미지 단독 평균 MAPE: raw 9.354773, 기존 standard SSL 9.292635, low_cov 9.311844, spatial_anchor 9.335322. 두 새 변형 모두 사전 지정 연구 gate 미달, 자동 승격 없음. Job 실행 경과 25,169.44초(약 6시간59분29초), 4 GPU × 경과 약 27.966 GPU·시간(스케줄링/종료 오버헤드 제외). GPU Pod는 정상 종료했다. 다음 DINO 실험은 `DINO_ALL9000_20260913.md`를 참조한다.

## 목적과 범위

사용자 요청: 단일 모델에서 SSL로 표현 능력을 더 개선한다. 여기서 단일 모델은 추론용 CPSAM encoder 하나와 기존 CD/CV/HEX head를 뜻한다. 기존 40-view TTA는 유지하며 앙상블은 평가·선택하지 않는다. SSL의 EMA teacher와 spatial anchor는 학습에만 쓰이며 추론 모델을 늘리지 않는다.

기존 제출물과 raw-label 제출용 압축파일은 변경하지 않는다. 새 실험은 fold 연구이며 전체 데이터 재학습, 배포, 공식 제출, Git push를 자동 실행하지 않는다. 모델 연산 및 테스트는 K8s에서만 수행한다.

## 변경 두 가지

| 설정 | 기존 standard SSL에서 바뀌는 것 | 검증하려는 가설 |
|---|---|---|
| low_cov | covariance penalty 계수 0.01 → 0.001 | 서로 관련된 세포 특징을 지나치게 분리하는 제약을 완화하면 회귀 표현이 개선되는가 |
| spatial_anchor | 기존 목적함수에 0.1 × frozen original CPSAM spatial-relation loss 추가 | 도메인 적응 중 유용한 공간 관계가 흐트러지는 것을 억제하면 개선되는가 |

두 변경을 동시에 적용하지 않는다. 공간 관계 손실은 원래 CPSAM의 고정 특징과 온라인 특징 사이 visible-token cosine Gram의 차이다. 마스킹된 위치 및 대각선은 제외한다. 원래 CPSAM의 공간 관계가 실제 세포 구조 정답이라는 보장은 없다.

분산·공분산 제약은 [VICReg](https://arxiv.org/abs/2105.04906), 공간 관계 보존 가설은 [DINOv3](https://arxiv.org/abs/2508.10104)의 관련 아이디어를 참고했다. **이 코드는 VICReg/DINOv3 공식 구현 또는 재현이 아니다.** 새 DINO 가중치나 외부 데이터는 사용하지 않는다.

## 고정 조건

- original CPSAM에서 각 실험 독립 초기화, SSL seed 20260911.
- native 256 crop, full_quality crop 선택 정책 그대로. 확대·축소 없음, gain ±5%, offset ±0.02만 사용.
- masked EMA SSL: 6,000 updates, batch 32, LR 1e-5, warmup 200, cosine floor 0.05, weight decay 0.05, mask 40–60%, variance coefficient 0.1, EMA 0.996→0.9999.
- 사전학습에 해당 fold 검증 이미지 사용 금지. 8,954장 raw pixel fingerprint 재검증, slide 및 exact-duplicate 중복 제거 split 고정. 환자 식별 및 near-duplicate 전체 문제까지 해결했다는 뜻은 아니다.
- fold 2: train 7,153, validation 1,791, primary 1,781. fold 3: train 7,161, validation/primary 1,791.
- 사전학습 이후 새로운 기존 구조 head, raw labels, seed 77, K=2, batch 48, encoder LR 1e-5/head LR 1e-3, 5,215 updates 고정. 최저 validation checkpoint를 고르지 않는다.
- 첫 세 downstream 학습 batch 해시를 기존 standard SSL과 일치 검증한다.
- train-only probe 32개, 500 update마다 feature std/effective rank 검사. 이는 붕괴 진단이며 과제 성능의 대체 지표가 아니다.
- spatial_anchor는 추가 frozen encoder forward가 있으므로 동일 업데이트 수지만 동일 연산량은 아니다.

## 평가 기준

기존 raw 및 standard SSL과 **동일 fold 2/3 primary 3,572장**에서 단독 모델 CD/CV/HEX MAPE와 평균을 비교한다. 기존 3-fold 5,350장 결과와 수치를 직접 비교하지 않는다.

사전 지정 연구 통과 기준: 기존 standard SSL 대비 평균 MAPE가 최소 0.03 낮고, fold 2와 3 각각에서도 개선. 이는 운영상 선별 기준이지 통계적 유의성 선언이 아니다. fold들은 이미 연구에 재사용되었으므로 독립 검증 또는 공식 점수 예측으로 해석하지 않는다. 통과해도 자동 제출/전체 재학습은 없다.

## 실행과 비용 상한

- Job: `clear-ec-representation-ssl-0913`, namespace `p-ajou-mlxp-project`.
- 생성: 2026-09-13 14:49 UTC / 10:49 EDT.
- Pod: `clear-ec-representation-ssl-0913-5jhg4`, node `h200-04-w-11f1`.
- H200 4장, 24 CPU, memory request 96 GiB / limit 192 GiB. 기존 PVC 사용.
- GPU 0/1: fold 2 low_cov/spatial_anchor. GPU 2/3: fold 3 low_cov/spatial_anchor. 각 GPU가 독립 모델 한 개를 처리한다.
- 각 worker는 SSL 4 updates → downstream 2 updates → 4-image 40-view 평가 및 strict reload smoke 성공 후 자기 본 학습으로 넘어간다.
- 본 실험 controller 10시간 상한, 최종 CPU 집계 최대 15분, 외부 timeout 10시간 15분, Job deadline 11시간(스케줄링 포함), 재시도 없음. 완료 후 TTL 10분으로 Pod 삭제/GPU 반환.
- 정상 완료 예상 초기값: 약 7–8시간. 기존 회귀 실측 약 2.97초/update × 5,215 ≈ 4.30시간. SSL 및 실제 이번 측정은 아래 launch snapshot으로 갱신한다.
- 4 GPU가 묶여 있으므로 일찍 끝난 worker GPU도 다른 worker 완료까지 할당된다. 예상 할당량 약 28–32 GPU·시간, controller 상한 약 40 GPU·시간에 종료/집계 오버헤드가 더해진다. 요금표가 없으므로 금액 추정은 하지 않는다.
- 첫 묶음 종료 후 자동 추가 실험은 없다.

## 검증과 위치

K8s CPU preflight: 6개 테스트 통과(관계 loss identity, student gradient/teacher detach, all-hidden, hidden reference 무시, objective 차이, gate). 기존 입력 및 원본 이미지 해시와 split 무결성 검증 완료. GPU Job server dry-run 통과.

원격 루트: `/workspace/clear-ec-research/experiments/representation_ssl_20260913/`

- `source/`: 실행 전 고정한 코드 3개.
- `manifest.json`: 소스/입력 해시, split, 설정, 사전 지정 gate.
- `logs/`, `status.json`, `complete.json`, 실패 시 `interrupted.json`.
- `smoke/{variant}_fold{2,3}/complete.json`: GPU smoke 완료 기록.
- `runs/{variant}_fold{2,3}/pretrain/`: SSL checkpoint, diagnostics, progress.
- `runs/{variant}_fold{2,3}/downstream/`: 고정 terminal checkpoint, config, split, OOF, result.
- `screen_audit.json`: 최종 집계와 연구 gate. 아직 생성 전이면 결과가 없는 상태다.

로컬에는 `results/representation_ssl_20260913/`에 집계 JSON만 복사한다. 이미지, 라벨, OOF 배열, 개별 예측, 새 체크포인트는 로컬로 내보내지 않는다.

OpenAI Docs 스킬을 사용해 기존 후속 확인 작업 `clear-ec-ssl`을 이 실험으로 갱신했다. 30분 간격 확인, 의미 있는 변화/완료/실패만 알림, 최종 보고 후 중지한다. 실제 시각 2026-09-14 14:49 UTC 이후 첫 확인을 최종 기한으로 둔다. 앱 후속 확인이 지연되더라도 K8s 학습 제한과 자동 종료는 독립적으로 적용된다.

## Launch snapshot

14:49 UTC: Pod Running, 4개 GPU smoke 시작.

14:51:30 UTC: 네 smoke 모두 성공하여 본 학습 진입, 네 실험 모두 SSL 50/6,000 updates 이상 확인. controller failed는 비어 있음. 처음 50 update 평균 low_cov 1.219–1.221초/update, spatial_anchor 1.464–1.465초/update. 이 속도를 단순 외삽하면 SSL 약 2.03/2.44시간, 회귀 약 4.30시간, 평가 및 저장 여유를 더해 전체 약 7–8시간(9월 13일 21:49–22:49 UTC / 17:49–18:49 EDT) 예상이다. 아직 초기 측정이며 장기 처리량·진단·I/O에 따라 변한다.

같은 순간 GPU 사용률은 29/44/91/82%, 프로세스 메모리 약 20.4/23.0/20.4/23.0 GB로 관측됐다. 순간 샘플을 지속 사용률로 주장하지 않는다. batch와 데이터 순서를 바꾸어 메모리를 채우지 않고 비교 조건을 유지한다. CPU preflight용 임시 Pod는 학습과 분리되어 정리한다. 아직 성능 개선 결과 없음.

## 20:20 UTC 중간 확인

실제 확인 시각은 2026-09-13 20:20 UTC이며 자동화 트리거의 17:03 UTC가 아니다. 네 실험 모두 SSL 6,000 updates 완료 후 원본 라벨 회귀 학습 중이다. SSL 실측 low_cov fold2/3 7,330.88/7,356.45초, spatial_anchor fold2/3 8,833.58/8,830.64초. 네 실행 모두 validation_images_used=0, 최종 probe 진단을 통과했다. 이 진단은 붕괴 검사용이지 검증 성능 향상의 증거가 아니다.

회귀 진행은 low_cov fold2/3 4,175/4,150 of 5,215, spatial_anchor 두 fold 3,700 of 5,215. 회귀 처리 속도 2.97–2.99초/update. 실패 없음, GPU 사용률 순간 표본 99–100%. 느린 두 실행은 회귀 약 75분이 남았고 이후 평가·저장·집계가 필요하므로 전체 결과까지 약 1.5–2시간 예상. 아직 최종 과제 점수는 없음.
