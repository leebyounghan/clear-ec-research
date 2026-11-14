# tier p3 / p3b — 결정 후 절차 (배포 레인, 2026-11-14 10:36 EST 작성; 시각은 파일 기록 시각)

팟에서 준비와 동등성 시험만 했습니다. 박스 전송·빌드·업로드·Colab 실행은 하지 않았고, p·t3·p2는 건드리지 않았습니다. 지시가 있을 때까지 아무것도 옮기지 않습니다.

## 두 변형
| | 바뀌는 것 | 상태 |
|---|---|---|
| **p3** (대비책) | p2에서 ViT-L s46 가중치를 인코더 LR 2e-5 전체 데이터 학습본으로 교체 | 준비·동등성 완료 |
| **p3b** | p3에 더해 ViT-L s47도 2e-5 학습본으로 교체 | s47 학습이 끝나고(예상 12:30-14:30) CD 레인이 상수를 올리면 같은 방식으로 준비합니다 |

- ViT-L은 CD 스택에서만 읽습니다. 그래서 두 변형 모두 CV·HEX는 p2와 같고, CD만 바뀝니다.
- 이미지 안에서 바뀌는 것은 매니페스트와 CD 상수뿐입니다. 모델 묶음은 tier p 묶음에서 ViT-L 파일 1개(p3) 또는 2개(p3b)만 바뀝니다. fp16 사본이며 파일당 0.61 GB입니다.
- **p3 CD 상수**: `freeze_cd_params_p3_20261114.json` (md5 `07273a5e…`).
  - strict CD는 6.1608입니다. p 6.1737보다 -0.0128 (CI -0.023 ~ -0.003), p2 6.1723보다 -0.0115 (CI -0.021 ~ -0.002) 낮습니다(CD 레인 O-CD-STAGE1-P3).
- **p3 팟 동등성** (스모크 5프레임, fp16b1, p2와 같은 조건, 같은 시각에 p2를 다시 돌려 대조):
  - CV·HEX: p2와 최대 4.5e-9 / 9.0e-10 차이입니다. p2를 두 번 돌렸을 때의 차이(2.0e-9 / 1.5e-9)와 같은 GPU 노이즈이며, 노이즈 외에 달라진 부품은 V뿐입니다.
  - CD: 5프레임 모두 p2보다 낮습니다(-0.11% ~ -0.94%, 최대 -26 cells/mm²).
  - ViT-L 단계: 4.89초로, 같은 시각 p2의 4.78초와 같습니다. 읽는 양도 같아서 예상 T4 시간은 p2와 같습니다(빠른 VM 258-272초, 느린 VM 278-296초).
- **참고**: 2e-5 전체 데이터 가중치 파일에는 HEX 정규화 통계가 NaN으로 저장돼 있습니다. 러너가 학습에는 nanmean을 쓰고 저장할 때만 mean을 써서 생긴 일입니다. V는 CD 채널만 쓰므로 영향이 없습니다. s47 파일도 같은 방식으로 확인합니다.

## 절차 (한 벌; 아래는 p3b 기준, p3로 가실 때는 명령의 `p3b`를 `p3`로 바꾸시면 됩니다)

### 1. 팟에서 직접 실행 (한 번, 약 1-3분 추정)
```bash
rsync -a --exclude test/ -e "ssh -i /root/.ssh/buildbox_20261112" /workspace/clear-ec-research/experiments/lane_cellroute_20261109/package_20261114/p3b/ qudgks@210.107.192.239:workspace/clear-ec/package_20261114/p3b/
```
다음 것들이 함께 갑니다.
- 빌드 컨텍스트(약 9 MB)
- 새 fp16 가중치(p3는 1개 0.61 GB, p3b는 2개 1.21 GB)
- 교체 목록 `model_swap.tsv`
- 박스 스크립트와 노트북 스크립트

### 2. 박스에서 (약 45-60분; 직접 돌리셔도 되고 배포 레인에 맡기셔도 됩니다)
```bash
bash ~/workspace/clear-ec/package_20261114/p3b/box/build_test_p3_20261114.sh
```
| 단계 | 하는 일 | 시간 |
|---|---|---|
| 빌드 | `clear-ec:<빌드 날짜>-freeze-9687076-p3b`. latest 안 씀, 같은 태그가 있으면 멈춤 | 약 2-5분 |
| 모델 트리 | `test_model_p3b` = 기존 `test_model_p` 파일을 하드링크로 잇고, 옛 ViT-L만 빼고 새 파일을 넣음(복사 없음, `test_model_p`는 그대로). 파일별 sha256 목록과 노트북용 패치 폴더 `upload_20261114/p3b_model_patch/` 생성 | 약 2분 |
| 테스트 | p와 같은 조건(RTX 4090, 네트워크 없음, 16g, uid 10001, 잡마다 새 컨테이너; Colab 5프레임 + 디버그 3건 × default/t4path + 쓰레기 파일 폴백). GPU를 쓰는 중이면 시작 안 함 | 약 35-45분 |
| 비교 | p 박스 테스트와 같은 잡끼리 비교. CV·HEX는 상대 1e-6 안이어야 함(GPU 노이즈 허용), CD 차이와 초 출력 | 1분 이내 |
| 이미지 저장 | `upload_20261114/clear-ec_<날짜>-freeze-9687076-p3b.tar.gz`(약 4.1 GB) + `SHA256SUMS_p3b.txt` | 약 3-5분 |

### 3. 노트북으로 받기 (패치 0.61 GB 또는 1.21 GB, 이미지 약 4.1 GB; 시간은 회선에 따라 다름)
```bash
mkdir -p ~/Downloads/clear-ec-upload_20261114
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/p3b_model_patch ~/Downloads/clear-ec-upload_20261114/
```
```bash
rsync -a --partial --progress "qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/clear-ec_*-freeze-9687076-p3b.tar.gz" ~/Downloads/clear-ec-upload_20261114/
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/SHA256SUMS_p3b.txt ~/Downloads/clear-ec-upload_20261114/
```
```bash
cd ~/Downloads/clear-ec-upload_20261114 && shasum -a 256 -c SHA256SUMS_p3b.txt
```

### 4. 노트북에서 모델 묶음 다시 만들기 (약 10분, 빈 공간 약 16 GB; 지금 75 GB 비어 있음)
```bash
python3 ~/Downloads/clear-ec-upload_20261114/p3b_model_patch/rebuild_model_archive_20261114.py ~/Downloads/clear-ec-upload_20261113/model_freeze-9687076-p_20261112.tar.gz ~/Downloads/clear-ec-upload_20261114/p3b_model_patch ~/Downloads/clear-ec-upload_20261114/model_freeze-9687076-p3b_20261114.tar.gz
```
스크립트가 하는 일:
1. 기존 p 묶음 sha256이 `fff3ec09…`인지 확인합니다(약 10초).
2. 새 파일의 sha256을 `model_swap.tsv`와 대조합니다.
3. 디스크에 풀지 않고 한 번에 새 묶음을 씁니다(약 8분).
4. 새 묶음을 다시 읽어, 파일별 sha256·크기·경로가 박스가 테스트한 트리 목록과 한 줄도 다르지 않은지 봅니다. 모든 파일이 읽기 가능한지, `._` 같은 macOS 파일이 없는지도 확인합니다(약 1분).

마지막 줄에 `CONTENTS MATCH`가 나와야 합니다. 다르면 `.part`로 남고 이름이 바뀌지 않습니다. 새 묶음 자체의 sha256은 `.sha256` 파일에 적힙니다. 묶음 해시는 p와 다르며, 그게 정상입니다.

이 과정은 이 Mac에서 실제 p 묶음으로 읽기 시험(65초, 443개 파일 이상 없음)과 압축 속도 측정(33 MB/s)을 했고, 작은 GNU tar 묶음으로 전체를 시험했습니다(맞으면 통과, 틀린 기본 묶음과 내용 불일치는 거부).

### 왜 노트북 재구성인가 (권장)
- **(a) 박스에서 새 묶음을 통째로**: 박스에서 `build_test_p3_20261114.sh archive`(약 3-5분)를 돌린 뒤 약 15 GB를 내려받습니다. 11-13 다운로드는 서버 타임아웃으로 재시도가 필요했고 22:31에 끝났습니다.
- **(b) 노트북 재구성(권장)**: 0.61-1.21 GB만 받고 노트북에서 약 10분 걸립니다.
  - 검증은 묶음 해시가 아니라 내용(파일별 sha256)으로 합니다. Grand Challenge가 풀어서 컨테이너에 주는 것이 바로 그 내용이므로 더 직접적인 확인입니다.
  - 기존 p 묶음이 온전한지도 처음에 확인합니다.
- 10 GB 업로드 한도 문제는 p와 똑같이 남아 있습니다(새 묶음도 약 15.0 GB).
