# CLEAR-EC 개발 기록·재현 자료 위치 안내

2026-09-15 갱신. 최신 제출·연구 상태는 [STATUS_20260915.md](STATUS_20260915.md)를 먼저 보세요.
이 문서는 **분산된 기록의 찾아보기**이며, 전체 재학습 재현이 검증됐다는 뜻은 아닙니다.
기존 문서를 이동하거나 학습 중인 코드·모델을 변경하지 않았습니다.

## 1. 저장 위치와 GitHub 반영 상태

- GitHub: <https://github.com/leebyounghan/clear-ec-research> (비공개).
- Mac 저장소: `/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research`.
- Kubernetes 프로젝트 루트: `/workspace/clear-ec-research`.
- Kubernetes namespace: `p-ajou-mlxp-project`; PVC: `clear-ec-workspace`; 마운트: `/workspace`.
- 이번 업데이트 이전 원격 기준: `ac99e27ee94142dd705950cfbe9901688411fbd2`.

이하 상대 경로는 해당 저장소 루트 기준입니다. Kubernetes 경로는 Mac 경로가 아닙니다.
파드는 종료·교체될 수 있지만 PVC의 데이터는 별도로 보존됩니다.

| 구분 | 확인된 상태 |
|---|---|
| `review_submission_20260910/` 코드·문서·PDF·설정·해시 | GitHub에 있음. 대형 가중치 자체는 제외 |
| `code/`, 날짜별 `docs/`, `k8s/` 실행 기록 | 이번 연구 저장소 업데이트 대상. 이력 보존이며 모든 실험 재실행 인증은 아님 |
| 최근 `results/` 실험 요약 | 선별된 집계·해시 자료만 반영. 이미지별 결과·전체 PVC 미러는 제외 |
| 원본 이미지·캐시·실험별 가중치·개별 영상 OOF 예측 | 주로 Kubernetes PVC. GitHub 배포 대상이 아님 |
| 이 `DEVLOG_INDEX.md` | 이번 업데이트에 포함되는 인덱스 |

## 2. September 10 제출본: 먼저 볼 자료

September 14의 최신 8.7010 결과와 SSL 여섯 번째 모델 변경은
[SSL_RELEASE_20260914.md](SSL_RELEASE_20260914.md)에 별도로 기록합니다.
아래 PDF와 review 폴더는 September 10 고정본이며 최신 SSL 방법론 PDF가 아닙니다.

| 목적 | 위치 |
|---|---|
| 설치·추론·Docker 사용법 | [review README](../review_submission_20260910/README.md) |
| 방법론 PDF | [Technical_Report.pdf](../review_submission_20260910/output/pdf/Technical_Report.pdf) |
| 6개 모델의 재학습 조건·명령·남은 검증 과제 | [TRAINING.md](../review_submission_20260910/TRAINING.md) |
| 고정된 6모델 구성·보정 계수 | [candidate6_full_quality.json](../review_submission_20260910/configs/candidate6_full_quality.json) |
| 제출 추론 코드 | [app/](../review_submission_20260910/app/) |
| 제출본 학습 코드 | [training/](../review_submission_20260910/training/) |
| 각 모델의 seed·선택 step·학습 조건 | [provenance/training/](../review_submission_20260910/provenance/training/) |
| 아카이브·가중치·소스 해시 및 실행 환경 기록 | [provenance/](../review_submission_20260910/provenance/) |
| 모델 배포 안내·아직 필요한 다운로드 링크 | [checkpoints/README.md](../review_submission_20260910/checkpoints/README.md) |
| 관리자 전달 전 체크리스트 | [RELEASE_CHECKLIST.md](../review_submission_20260910/RELEASE_CHECKLIST.md) |

공식 결과는 [8.7042 평가 페이지](https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/).
최신 README에 성공한 평가·이미지·모델 ID가 기록돼 있습니다. 체크리스트의 해당 항목은 아직 예전 미확인 상태여서 README와 대조해야 합니다.
PDF는 기존 방법론 문서를 보존한 것이며, 최신 공식 점수는 README에 추가됐습니다.

## 3. 날짜별 개발 기록: devlog 역할을 하는 문서들

아래는 **저장소 문서 링크**입니다. 날짜는 문서/실험 식별자이며, 그날 모든 학습이 완료됐다는 뜻은 아닙니다.
각 문서의 이후 업데이트와 완료 근거를 함께 읽어야 합니다.

| 시기 | 내용 | 문서 |
|---|---|---|
| 초기 | 기본 회귀·앙상블·실패 실험과 한계 | [FINDINGS.md](FINDINGS.md), [METHODOLOGY.md](METHODOLOGY.md) |
| 09-02 | 초기 결론 재검토·학습 구현 문제 후속 확인 | [FOLLOWUP_20260902.md](FOLLOWUP_20260902.md) |
| 09-04 | 원본9,000장·ID 정합·제출 후보 | [SUBMISSION_20260904.md](SUBMISSION_20260904.md), [FINAL_CANDIDATE_20260904.md](FINAL_CANDIDATE_20260904.md) |
| 09-04~05 | 손실·crop bag·추가 모델 비교 및 학습 | [METHODS_20260904.md](METHODS_20260904.md), [DEPLOY_METHODS_20260905.md](DEPLOY_METHODS_20260905.md) |
| 09-05 | 초기화 수정·제출 추론 수치 검사 | [DEPLOY_REVIEW_20260905.md](DEPLOY_REVIEW_20260905.md) |
| 09-05~07 | 원본 해상도 영역 선택·seed 확인·regional 후보 | [REGION_SEARCH_20260905.md](REGION_SEARCH_20260905.md), [REGION_SEEDS_20260906.md](REGION_SEEDS_20260906.md) |
| 09-08~09 | 지표별 specialist·통계 잔차 비교 | [SPECIALIZE_SCREEN_20260908.md](SPECIALIZE_SCREEN_20260908.md), [STATRES_SCREEN_20260908.md](STATRES_SCREEN_20260908.md) |
| 09-09~10 | 두 번째 제출본 패키징·환경 검사 | [PACKAGING_20260909.md](PACKAGING_20260909.md) |
| 09-10~11 | crop 후속 비교·DINO 검토 | [CROP_DINO_SCREEN_20260910.md](CROP_DINO_SCREEN_20260910.md) — 문서에 등장한다고 모두 실행된 것은 아님 |
| 09-11 | 제출 후 방향·추가 사전학습 비교 | [LEADER_FOLLOWUP_20260911.md](LEADER_FOLLOWUP_20260911.md), [SSL_SCREEN_20260911.md](SSL_SCREEN_20260911.md) |
| 09-12 | 원본/평균화 라벨 비교·다른 fold 확인 | [TARGET_SCREEN_20260912.md](TARGET_SCREEN_20260912.md), [TARGET_REPLICATION_20260912.md](TARGET_REPLICATION_20260912.md) |
| 여러 시기 | 외부 데이터 출처·사용 판단 | [EXTERNAL_DATA.md](EXTERNAL_DATA.md) |
| 09-13 | 원본 라벨 + standard SSL 및 representation 비교 | [RAW_SSL_NEXT_20260913.md](RAW_SSL_NEXT_20260913.md), [REPRESENTATION_SSL_20260913.md](REPRESENTATION_SSL_20260913.md) |
| 09-14 | CPSAM DINO/I-JEPA 및 SSL 길이 비교 | [CPSAM_METHODS_20260914.md](CPSAM_METHODS_20260914.md), [SSL_DURATION_20260914.md](SSL_DURATION_20260914.md) |
| 09-14~15 | 제출용 sixth-only SSL·나머지 5개 재학습·증류 계획 | [SSL_RELEASE_20260914.md](SSL_RELEASE_20260914.md), [ALL_MEMBERS_SSL_20260915.md](ALL_MEMBERS_SSL_20260915.md), [DISTILLATION_NEXT_20260915.md](DISTILLATION_NEXT_20260915.md) |

`FINDINGS.md`의 일부 오래된 표·`METHODOLOGY.md`·[REPRODUCE.md](REPRODUCE.md)는 초기 방식 설명이 남아 있습니다.
September 10은 review 패키지, 이후 제출은 최신 상태·release 문서로 구분하세요.
이후 연구를 제출에 사용된 방법처럼 합치면 안 됩니다.

## 4. 실험 원본: Kubernetes에서 찾는 위치

아래 폴더는 `/workspace/clear-ec-research/experiments/` 아래 존재함을 확인했습니다.

| 내용 | 하위 폴더 |
|---|---|
| 초기 후속 진단 | `followup_20260902/` |
| 3개 seed 최종 fit | `final_candidate_20260904/` |
| 방법론 비교·최종 fit·수치 검사 | `methods_20260904/`, `deploy_methods_20260905/`, `deploy_review_20260905/` |
| native 영역 탐색 | `region_search_20260905/` |
| seed 검증 | `region_seedcheck_20260906/`, `region_seedconfirm_20260906/` |
| regional 최종 fit·후보·추론 검사 | `deploy_region_20260907/`, `region_candidate_20260907/`, `region_inference_20260907/` |
| specialist·통계 잔차 | `specialize_screen_20260908/`, `statres_screen_20260908/` |
| crop 후속·SSL | `crop_dino_screen_20260910/`, `ssl_screen_20260911/` |
| 라벨 비교 fold0/1 | `target_screen_20260912/` |
| 라벨 비교 fold2/3 및 fold4 | `overnight_target_20260912/`, `target_fold4_20260912/` |

실험마다 구조가 조금씩 다르지만 다음 파일을 우선 확인합니다.

- `source/`: 그 실험에 고정된 코드. 저장소 최상위 `code/`보다 당시 실행 근거로 우선합니다.
- `manifest.json`: 입력·소스 해시, 실험 구성, split 등.
- `runs/.../config.json`, `progress.json`, `result.json`: 학습 설정·진행·결과.
- `split_indices.npz`, `oof.npz` 또는 `best_oof.npz`: 분할과 개별 예측. 외부 공유 전 별도 검토 필요.
- `selected.pt` 또는 `best.pt`: 실제 체크포인트. 파일명/선택 기준은 실험별 확인.
- `screen_audit.json`, `complete.json`, `queue_finished.json`: 비교·감사·완료 근거. 모든 옛 실험에 동일 이름으로 존재하는 것은 아닙니다.
- `logs/` 또는 프로젝트 루트 `logs/<실험명>/`: 실행 로그.

폴더 존재나 `Running`만으로 성공을 판단하지 마세요. `failed`, `audit_failed.json`, `interrupted.json`과 최종 결과를 함께 확인합니다.
기록에 적힌 예전 파드 이름·진행 상태는 현재 상태 보증이 아닙니다.

## 5. Mac에 복사된 결과 요약

저장소의 [results/](../results/) 아래에 다음 요약 복사본이 있습니다. 전체 PVC 미러는 아닙니다.

- `methods_20260904/`: 방법론 비교 결과.
- `crop_screen_20260911/`: crop 비교와 완료·감사 기록.
- `ssl_screen_20260911/`: 사전학습 비교와 완료·감사 기록.
- `target_screen_20260912/`: 현재 로컬에는 시작 당시 manifest·launch/GPU·target audit만 있음.
  **최종 `screen_audit.json`은 이 로컬 폴더에 아직 없으므로 PVC 원본을 확인해야 합니다.**
- `final_candidate_20260904_*`, `submission_20260904_ensemble_audit_final.json`: 이전 후보·보정·앙상블 검사.
- `docs/SPECIALIZE_SCREEN_20260908_audit.json`, `docs/STATRES_SCREEN_20260908_audit.json`: 두 비교의 감사 요약.

## 6. 데이터·가중치·제출 압축파일

| 항목 | 위치 | GitHub |
|---|---|---|
| 원본 학습 압축파일 | Kubernetes `/workspace/artifacts/clear-ec-train9000-raw.tar.gz` | 없음 |
| 기존8,954장 중앙 cache·메타데이터 | Kubernetes `/workspace/clear-ec-research/datasets/clear_ec_official/` | 데이터 없음 |
| 원본 전체 해상도·영역 품질 cache | Kubernetes `/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/` | 없음 |
| 9,000장 ID 정합 후 최종 cache | Kubernetes `/workspace/clear-ec-research/datasets/clear_ec_final9000_20260904/` | 없음 |
| 공개 foundation checkpoint | Kubernetes `/workspace/clear-ec-research/cpsam_models/cpsam_v2` | 가중치 없음 |
| 제출 6모델 staging | Kubernetes `/workspace/clear-ec-research/experiments/region_candidate_20260907/experimental_stage/model/` | 가중치 없음 |
| 모델 압축파일·Docker 압축파일·해시 | Mac 저장소 `submission_candidate_20260909/artifacts/` | 대형 파일 없음 |

실제로 보관된 제출 압축파일은 다음 두 개입니다. 새 실험 결과로 덮어쓰지 마세요.

- `submission_candidate_20260909/artifacts/model_full_quality77_20260909.tar.gz`
- `submission_candidate_20260909/artifacts/clear_ec_full_quality77_20260909.tar.gz`
- 같은 폴더의 `SHA256SUMS.txt`와 review 패키지 `provenance/`로 동일성을 확인합니다.

## 7. 재현할 때의 권장 읽기 순서

1. **제출 모델로 추론:** review README → 모델 배포 안내 → 체크섬 검사 → 고정 추론 실행.
2. **제출 모델 재학습:** review TRAINING → `provenance/training/` → `training/` 코드.
3. **왜 그 방법을 선택했는지:** 위 날짜별 문서 → 대응 실험의 frozen source·split·결과.
4. **남은 재현 검증:** 학습 환경 고정, 원본 데이터 bootstrap, OOF 선택·보정 과정 연결,
   깨끗한 Kubernetes 작업공간에서 전 과정 실행 및 추론 수치 비교.

이번 업데이트는 연구 코드·문서와 선별된 집계 자료의 Git 반영입니다.
데이터/모델 공개 다운로드 경로 생성이나 완전한 재학습 재현 검증은 별도입니다.
