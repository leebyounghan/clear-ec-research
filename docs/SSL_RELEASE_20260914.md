# 제출용 standard SSL + 원본 라벨, 전체 데이터 — 2026-09-14

## 사용자 요청과 의미

기존 비교에서 가장 좋았던 standard masked EMA SSL + raw labels를 **전체 데이터로 학습하여 모델 전용 제출 압축파일을 만든다**. DINO나 새 representation 변형을 사용하지 않는다. 공식 업로드·제출·Git push는 하지 않는다.

기존 6개 앙상블의 앞 5개 `seed42.pt`, `seed77.pt`, `seed2026.pt`, `metric35.pt`, `predbag2_35.pt`는 재학습하지 않고 보존한다. 6번째 `full_quality_seed77.pt`만 이번 결과로 교체한다. 최종 예측은 6개 예측의 동일 가중치 평균에 CD/CV/HEX별 기존 보정계수를 곱한다:

| CD | CV | HEX |
|---:|---:|---:|
| 0.9970446110788349 | 0.987972130802936 | 0.9892234872488541 |

모델 가중치와 이 보정계수는 별개다. 새 6번째 모델에 보정계수를 재최적화하지 않는다. 기존 fold 비교와 교체 범위를 맞추기 위한 선택이지 새 조합에서 보정계수가 최적이라는 뜻은 아니다. 기존 두 제출 패키지와 raw-only 패키지를 덮어쓰지 않는다.

## 고정 학습

- SSL: CPSAM original initialization, **9000장 전체**, label-free, 6,000 updates, batch32, LR1e-5, 기존 `ssl_native_core.py` 그대로. Masked EMA prediction + variance0.1/covariance0.01. Native256 full_quality, resize 없음. SSL train 호출만 기록하여 마지막에 실제 unique sampling 9000 확인; RNG나 학습 순서는 바꾸지 않는다.
- 회귀: SSL terminal EMA encoder + 새 기존 구조 head, **유효 원본 라벨 8998장**(비양수 라벨 2장 제외), seed77, K2/batch48, encoder1e-5/head1e-3, 5,215 고정 updates. 라벨 평균화 없음, validation-best 선택 없음.
- 전체 데이터 최종 학습이므로 독립 validation score는 없다. 이전 동일5350장 연구 비교에서는 raw-only fixed6 9.029185 vs standardSSL+raw fixed6 9.019599. 이 차이를 공식8.7042나8.6206의 개선 예상치로 사용하지 않는다.
- 모델 6개 전체를 SSL로 재학습하는 작업이 아니다.

## 실행과 검증

원격 `/workspace/clear-ec-research/experiments/ssl_release_20260914/`, namespace `p-ajou-mlxp-project`, PVC `clear-ec-workspace`.

- GPU Job `clear-ec-ssl-release-0914`: H200 **1장**, CPU6, request32Gi/limit64Gi. 단일 최종 학습이므로 불필요하게 여러 GPU를 예약하지 않는다. smoke(SSL4 + regression2 + 4-image40view/strictreload) → full SSL → regression → stage → isolated CUDA12.6 frozen-app runtime 순서.
- CPU Job `clear-ec-ssl-release-archive-0914`: GPU 없음, CPU2. 성공 marker를 기다리고 압축·8개 archive member hash 재검증. GPU는 runtime 이후 종료하여 압축 중 잡아두지 않는다.
- controller9h 상한, 외부9h5m timeout, GPU Job10h scheduling 포함, archive wait10h/Job11h, retry0, finished TTL10min. 실패 시 자동재학습 없음.
- 이전 검증된 helper/source를 그대로 import하고 경로만 isolated ROOT로 바꾼다. manifest는 source/input hash와 기존 archive 8개 항목 hash를 고정한다. 학습 도중 frozen source를 수정하지 않는다.
- stage는 앞5개+foundation+candidate config 해시 보존 및 sixth만 변경 확인.
- runtime은 기존 별도 venv `experiments/raw_ssl_next_20260913/runtime_setup/.venv/bin/python` (torch2.11 CUDA12.6) 사용. 8개 입력에서 앞5개 예측 동일성, FP16/BF16 drift<=0.5%, 출력 범위와 frozen app 실제 entrypoint 출력 확인. **공식 T4/실제 업로드된 Docker 검증은 아님**.
- 훈련·모델 로딩·검증·압축은 K8s에서 수행한다. Mac에서는 파일 편집/전송/해시검증만 수행한다.

## 상태 및 산출물

- `manifest.json`, `gpu_status.json`, `gpu_complete.json`, `interrupted.json`.
- `logs/{smoke,full_train,stage,runtime}.log`.
- `pretrain/{progress,result,coverage}.json`, `encoder_step6000.pt`.
- `raw_deploy/{config,result}.json`, `selected.pt` (SSL+raw지만 기존 stage helper 호환 디렉터리 이름 유지).
- `model/`, `stage_manifest.json`, `runtime_check/result.json`.
- `ready_for_archive.json`, `package_ready.json`.
- `artifacts/model_full_quality77_ssl_raw_20260914.tar.gz`, `SHA256SUMS.txt`, `model_archive.json`.

파일 생성은 runtime+archive 검증 이후에만 완료로 보고한다. 로컬 전달 위치는 `submission_candidate_ssl_raw_20260914/`. 모델 압축파일 다운로드는 사용자 요청 범위이며 허용된다. 개별 이미지/라벨/OOF/runtime 이미지별 예측은 로컬로 내보내지 않는다. 완료 archive와 집계/provenance만 전송하며 SHA256을 확인한다.

기존 `clear-ec-ssl` heartbeat를 이 제출 준비와 전달로 갱신한다(OpenAI Docs 사용). 30분 간격, 의미 있는 변화/실패/완료만 알림. 부재 시 PVC 검사에만 GPU없는 수명제한 읽기 Pod 허용. 완료 파일 로컬 검증 후 일시정지, 실패/새 권한 필요 시 보고 후 일시정지. 절대 후속 확인 마감은 실제2026-09-15 12:00UTC. 추가 학습/공식제출/Git push 금지.

## 초기 예상

18:34:27UTC 실제 확인: GPU Pod `clear-ec-ssl-release-0914-ghn58`, node `h200-04-w-11f1`, H200 1장. 기존 SSL CPU 테스트3/3 통과, SSL4→회귀2→4장 평가/strictreload smoke 통과 후 본 SSL25/6000 진행(1.227초/update). CPU archive Pod `clear-ec-ssl-release-archive-0914-4zbsp`는 성공 marker를 대기 중. 아직 완성 모델/압축파일 없음. 준비용 CPU Pod는 정리하며 기존 제출 파일은 유지한다.

9월14일18:32UTC 전후 준비 완료. 기존 실측 SSL ~2h + 회귀 ~4h15m + runtime/압축/전송 여유로 **약7–8시간 후 로컬 전달** 예상. 이는 9월15일01:30–02:30UTC, 9월14일21:30–22:30EDT, 9월15일10:30–11:30KST 부근이며 공식 마감시각과 별개다. 정확한 제출 마감 timezone은 이 작업에서 확인되지 않았다. GPU safety cap9h는 예상 소요시간이 아니다.

## 20:35UTC 확인 — 전체 데이터 SSL 완료

9월14일20:34:16UTC에 본 SSL6000/6000 updates 도달. 결과 기록의 학습 시간은
7221.98초(약2시간00분22초), peak torch memory16.39GiB다. Coverage는
eligible9000/unique9000/labels_used=false를 확인했다. 최종 encoder SHA256을
PVC에서 직접 다시 계산하여 결과 기록과 일치함을 검증했다:
`971bcaba9f66cbf9466b6e4bfd3e5e26f593ef56384dee1a2368133647bded32`.
집계 result.json과 coverage.json만 로컬 results/ssl_release_20260914/pretrain에 복사했다.

원본 라벨 회귀 config가 생성됐으며, train8998/updates5215/seed77 및 위 SSL
checkpoint SHA가 일치한다. 20:35:12UTC 확인 때 첫25-update progress는 아직
발행 전이므로 회귀 완료/실측 속도로 오해하지 않는다. GPU/CPU archive Pod 모두
Running, restart0. Source/input 무결성은 release.verify()로 재검증했다.
압축 준비 marker는 아직 없으며, 회귀와 runtime 검사 후 archive가 시작된다.
SSL representation diagnostics만으로 검증 정확도 개선을 주장하지 않는다.

독립 duration 비교는 두fold 모두 첫2000-step initializer의 downstream
2125/5215(20:31UTC 기록), 약2.978/2.980초/update였다. Duration manifest 및
source/input SHA도 재검증했다. 증류 선행조건은 아직 미충족이며 추가 GPU나
학습 재시작/제출물 변경은 하지 않았다. 기존 통합 heartbeat는 계속 유지한다.

## 사용자 확인 마감 및 21:17UTC 정기 점검

사용자가 플랫폼 문구를 직접 제공했다: Phase I은 **2026년9월15일06:00
America/New_York**, 즉 **2026년9월15일10:00UTC / 19:00KST**에 자동 종료된다.
앞선 마감 시간대 미확인 기록은 이 사용자 제공 정보로 보완한다. 제출용 파일은
충분한 업로드/플랫폼 확인 여유를 두고 먼저 전달하며, 공식 제출은 사용자 결정이다.
앞5개까지 교체할 시간에 관한 최근 질문은 가능성 검토였으며, 추가5개 학습이나
기존 연구/증류 우선순위 변경을 실행하도록 승인한 것으로 해석하지 않는다.

21:17:32UTC 실제 확인: 제출용 회귀850/5215(21:16:38UTC),2.984초/update;
duration 두fold는 각각3025/5215,2.984/2.987초/update로 첫2000 initializer의
회귀 학습 중이다. 네 기존 Pod 모두Running/restart0. 새 실패/완료/archive
marker 없음. 두 실행 소스와 duration manifest SHA는 고정값과 일치한다.
전체 입력 해시는20:35UTC 검증 기록이 최신이며 이번에는 재계산하지 않았다.
추가 GPU/재시도/패키징/증류 실행 없이 기존 학습을 유지한다.

## 9월15일01:17UTC — 원격 완성, 로컬 전송 중

전체 원본 라벨 회귀는00:51:06UTC에5215updates 완료(15402.14초,
약4시간16분42초), strict reload 통과. GPU controller는00:55:43UTC에
failed={}로 종료했다. CPU 압축/8개 member 해시 검증은01:02:33UTC 완료.
두 제출용 Job은 TTL로 삭제됐고 연구용 두GPU만 계속 학습 중이다.

원격 archive는 `model_full_quality77_ssl_raw_20260914.tar.gz`,
7,964,665,157 bytes, SHA256
`e37c83db830cc5fa5215f7fd63267beb11e6c09a8778aefe2b8c6180a37a9730`.
새6번째 checkpoint SHA256은
`537e20635164ab56600e3167f8a25baabc65b002d080cfe4ac143c451c546669`.
01:16–01:17UTC CPU helper에서 source/input 검증과 최종 encoder/회귀/
stage8개/압축파일 SHA를 다시 계산했다. 나머지7개 member는 기존 manifest와
완전히 동일하다(앞5개+foundation+config). SSL9000장 실제 sampling,
6000updates 및 raw8998/5215updates를 재확인했다.

Frozen CUDA12.6 app runtime8장 검사 통과. 앞5개 예측은 정확히 동일하며
FP16/BF16 최대 상대차0.00065146(약0.0651%). 실제 entrypoint도 통과했다.
공식T4 또는 실제 업로드 Docker 검증은 아니며 정확도 개선 검증도 아니다.
로컬에는 runtime 개별예측 대신 집계 `runtime_summary.json`만 저장한다.

로컬 여유공간 약12GiB를 확인하고 새
`code/fetch_ssl_release_20260914.py`로 모델만 전송 중이다. 기존 downloader와
기존 제출물은 수정하지 않았다. 8개 병렬64MiB range를 하나의 partial파일에
기록하여 이중8GB 디스크 사용을 피하며 전체SHA 확인 후에만 최종 이름으로
전환한다. 아직 로컬 전달 완료로 보고하지 않는다. CPU helper는 이번에 생성한
`clear-ec-target-preflight-0912`(GPU없음,1시간상한)이고 전송 후 정리한다.
Duration 두fold4000 initializer 회귀는 약2330/5215이며 증류 선행조건은
아직 미충족. 제출 완료와 무관하게 통합 후속 확인은 계속한다.

전송 보완: 기본 websocket 대용량 exec에서 EOF/timeout이 발생했다. 과거
range downloader와 동일한 `KUBECTL_REMOTE_COMMAND_WEBSOCKETS=false` 설정,
2MiB 소청크 및 stream drain2초로 전환했다. 16개 병렬 연결로64MiB 단위 완료를
기록하며 partial파일 하나만 쓴다. 이미 확인한 모델/원격 archive는 변경하지
않고 전송만 재개한다. 원격 검증과 로컬 완료 상태는 별도로 기록한다.

01:24UTC 전송은 계속 실행 중: 로컬 PID11235, exec session40419,
`python3 code/fetch_ssl_release_20260914.py`. Partial의 실제 할당량350MiB를
확인했다(파일표시7.4GiB는 sparse 예약 크기이며 수신완료량이 아님).
다음 확인에서는 먼저 이 프로세스/세션과 `download_verified.json`을 확인하고,
살아 있는 downloader와 중복 실행하지 않는다. CPU helper의 생성시각은
01:14UTC 전후이고 deadline3600초다. 전송 실패/기한 종료 시 원격 성공품은
그대로 유지하고, 완료range plan 및 동일 archive SHA를 확인하여 새CPUhelper로
전송만 재개할 수 있다. 완료 시 이번helper만 삭제한다. 최근 실측 수신속도로
약30–40분 추가 예상이나 네트워크 상황에 따라 달라진다. 사용자에게 아직
로컬 완성파일 링크를 전달하지 않았다.

01:56UTC 확인: 동일 downloader PID11235/session40419 정상 실행 중,
완료 기록된 range5,368,709,120/7,964,665,157bytes(약67.4%), 진행 중
청크 포함 partial 실제할당약5.4GiB. 최종파일/검증marker는 아직 없다.
로컬 여유6.9GiB로 남은분량 수신 가능. helper는생성41분경과,
기존3600초상한그대로이며 새helper/중복downloader를 시작하지 않았다.
다음 확인은 먼저 session40419 출력 및 download_verified.json을 확인한다.
완료됐으면 전체SHA 검증 기록/파일 확인 후 사용자에게 즉시 링크를 전달하고,
이 helper만 정리한다. 실패했으면 기존부분파일 및plan을 보존하고 원인을
확인한다. 연구두fold는4000 initializer3175/5215로 계속 실행 중이므로
증류는 아직시작하지 않으며 통합 heartbeat도 유지한다.

## 로컬 전달 완료 — 9월15일02:28UTC 확인

Downloader session40419는exit0으로 종료했고 최종 전체SHA 검증을 통과했다.
로컬 완료시각은download_verified.json에 기록(뉴욕9월14일22:09경).
최종파일은 `submission_candidate_ssl_raw_20260914/model_full_quality77_ssl_raw_20260914.tar.gz`,
7,964,665,157bytes, SHA256
`e37c83db830cc5fa5215f7fd63267beb11e6c09a8778aefe2b8c6180a37a9730`.
사용자의 긴급 제출 문의에 최종 경로를 전달한다. 다운로드/모델 재생성 금지.
이번 전송CPUhelper는Completed 상태 확인 후 정확한 이름으로 삭제했다.
기존연구GPU두개는Running이며 건드리지 않았다. 제출전달branch는완료,
duration감사와증류branch가남았으므로통합heartbeat는계속한다.
모델전용파일이며기존호환Docker와분리업로드한다. 공식플랫폼테스트/점수향상
검증과공식제출은아직수행하지않았다.
