# tier p3 업로드·Try-out (교수님용, 2026-11-14 14:31 EST 작성, p3b 추가 15:07 EST; 시각은 파일 기록 시각)

Phase II 제출 버튼은 이 절차에서 누르지 않습니다. p3b는 이 파일 맨 아래에 있습니다.

## 박스 테스트 결과 (RTX 4090, Grand Challenge 방식, 14:09-14:30 EST)
- 17개 작업이 모두 성공했습니다.
  - T4 경로 강제: 작업당 68-79초 (p는 같은 작업에서 70-79초).
  - 기본 경로: 75-86초.
  - RAM 최대 5.7 GB, GPU 최대 4.3 GB입니다.
- CV·HEX는 p와 같습니다. 차이는 상대 1.1e-8 이내로, GPU 노이즈 수준입니다.
- CD만 바뀝니다. 같은 프레임에서 p보다 -0.48 % ~ +0.16 % 차이가 납니다.
- 깨진 파일을 넣으면 폴백 상수(2770, 0.36, 0.53)가 나옵니다.

## 0. 파일
| 파일 | 크기 | sha256 (앞 16자) |
|---|---:|---|
| `clear-ec_20261114-freeze-9687076-p3.tar.gz` (이미지) | 4.11 GB | `697391c48591767e` |
| `model_freeze-9687076-p3_20261114.tar.gz` (모델, 노트북에서 재구성) | 약 15.0 GB | 재구성 마지막 줄과 `.sha256` 파일에 적힌 값 |

- 이미지 ID는 `c4ef95dfd359`, 태그는 `clear-ec:20261114-freeze-9687076-p3`입니다.
- 모델 묶음은 p 묶음에서 ViT-L 파일 1개만 바꾼 것입니다. 10 GB 업로드 한도 문제는 p와 똑같이 남아 있습니다.

## 1. 노트북으로 받기·재구성 (한 줄씩)
```bash
mkdir -p ~/Downloads/clear-ec-upload_20261114
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/p3_model_patch ~/Downloads/clear-ec-upload_20261114/
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/SHA256SUMS_p3_model_patch.txt ~/Downloads/clear-ec-upload_20261114/
```
```bash
cd ~/Downloads/clear-ec-upload_20261114/p3_model_patch && shasum -a 256 -c ../SHA256SUMS_p3_model_patch.txt
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/clear-ec_20261114-freeze-9687076-p3.tar.gz ~/Downloads/clear-ec-upload_20261114/
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/SHA256SUMS_p3.txt ~/Downloads/clear-ec-upload_20261114/
```
```bash
cd ~/Downloads/clear-ec-upload_20261114 && shasum -a 256 -c SHA256SUMS_p3.txt
```
```bash
python3 ~/Downloads/clear-ec-upload_20261114/p3_model_patch/rebuild_model_archive_20261114.py ~/Downloads/clear-ec-upload_20261113/model_freeze-9687076-p_20261112.tar.gz ~/Downloads/clear-ec-upload_20261114/p3_model_patch ~/Downloads/clear-ec-upload_20261114/model_freeze-9687076-p3_20261114.tar.gz
```
- 확인 결과: 패치 확인은 4줄 모두 OK, 이미지 확인은 1줄 OK가 나와야 합니다.
- 재구성은 약 10분 걸립니다. 마지막 줄이 `[5/5] CONTENTS MATCH the box-tested tree (443 files)`로 시작해야 합니다.
- rsync가 `poll: timeout`으로 멈추면 같은 줄을 다시 실행하시면 됩니다.

## 2. 업로드 (p와 같은 설정)
- Containers: **GPU NVIDIA T4**, **Memory 32 GB**. Comment는 `freeze 9687076 tier p3 20261114 sha 697391c4`입니다.
- Models: Comment는 `freeze 9687076 tier p3 model sha <재구성 sha 앞 8자>`입니다.
- 이미지와 모델은 반드시 같은 tier끼리 올립니다. p3 이미지에 p 모델을 붙이면 새 ViT-L 파일이 없어서 `model files missing`으로 실패합니다.

## 3. Try-out 기대값 (T4 경로 강제, 박스 4090 출력)
허용 범위는 CD ±0.1 %, CV ±0.5 %, HEX ±0.5 %입니다(11-13 p와 같은 기준).

| 파일 (`~/Downloads/`) | 이미지 | 기대 CD | 기대 CV | 기대 HEX |
|---|---|---:|---:|---:|
| `d4e8cdb0-765a-4b07-8091-d7da44ab6068.mha` | 0001-19 | 2859.5 | 0.3819 | 0.5462 |
| `58c72195-a5a7-48de-abea-02b3a3519a75.mha` | 0001-20 | 2004.0 | 0.4796 | 0.4134 |
| `767b441c-8a94-47e1-ad91-02f8757bcde9.mha` | 0001-21 | 4371.3 | 0.2977 | 0.6496 |

- 같은 파일에서 p는 CD 2863.2 / 2013.6 / 4364.3이었습니다. CV·HEX는 p와 같습니다.
- p2는 박스에서 빌드·테스트하지 않았습니다(p2 폴더가 박스에 오지 않았습니다). 그래서 p2 기대값은 없습니다.

## 4. 결과 확인 (Results → Logs)
| 항목 | 통과 기준 | 통과 못 하면 |
|---|---|---|
| Status | Succeeded | 로그를 전달 |
| Exec Duration | **270초 이하** | 넘으면 t3로 바꿉니다. t3 파일은 박스 `package_20261112/upload_20261113/`에 있고 노트북에는 없습니다 |
| 로그 마지막 줄 | `"mode": "recipe"`, `"errors": {}`, hooks 안에 `"t4_default": true`, `"autocast": "fp16"`, `"det_copies": 1` | `"mode": "constant"`면 폴백 상수가 나간 것이니 로그를 전달 |
| 출력값 | 위 표와 허용 범위 안 | 범위 밖이어도 recipe이고 errors가 비어 있으면 로그만 전달 |

- `t4_default`는 실제 T4에서만 찍힙니다. 박스의 강제 경로 로그에는 없습니다.
- hooks에는 `"ro_hits": 14`, `"read_bytes": 15393307648`도 보입니다(p는 24583744240).
- **tier 확인은 `decisions`의 마지막 항목으로 합니다.**
  - p3의 마지막 항목은 `"D-COORD-P3-PREP tier p3: cd = deploy block of experiments/lane_cd_20261108/reads/freeze_cd_params_p3_20261114.json (md5 07273a5e624551ae1b2e2ba194d8c2fd)"`입니다.
  - 첫 항목 `"D-COORD-P2-PREP tier p2 (...)"`는 p2 매니페스트에서 물려받은 기록이고, 오류가 아닙니다.

---

## p3b (박스 테스트 15:06 EST 완료): 절차는 위 p3와 같고, 이름의 `p3`만 `p3b`로 바꿉니다
- **결과**
  - 17개 작업이 모두 성공했습니다.
  - T4 경로 강제: 작업당 68-79초 (p3도 68-79초).
  - 기본 경로: 75-86초. RAM 최대 5.7 GB입니다.
  - CV·HEX는 p·p3와 같습니다(상대 1e-8 이내).
  - CD는 같은 프레임에서 p3보다 -1.10 % ~ +0.20 % 차이가 납니다.
  - 팟 동등성 시험도 통과했습니다.
- **파일**
  - 이미지: `clear-ec_20261114-freeze-9687076-p3b.tar.gz`, 4.11 GB, sha256 앞 16자 `67a8452f62fbced7`.
  - 이미지 ID는 `db390738df95`입니다.
  - 모델은 노트북에서 `model_freeze-9687076-p3b_20261114.tar.gz`로 재구성합니다.
- **패치 폴더 `p3b_model_patch/`**
  - ViT-L 파일이 2개 들어 있습니다.
  - s46 파일은 p3 패치와 같은 파일입니다.
  - 재구성의 OLD는 p3와 마찬가지로 p 묶음입니다.
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/p3b_model_patch ~/Downloads/clear-ec-upload_20261114/
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/SHA256SUMS_p3b_model_patch.txt ~/Downloads/clear-ec-upload_20261114/
```
```bash
cd ~/Downloads/clear-ec-upload_20261114/p3b_model_patch && shasum -a 256 -c ../SHA256SUMS_p3b_model_patch.txt
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/clear-ec_20261114-freeze-9687076-p3b.tar.gz ~/Downloads/clear-ec-upload_20261114/
```
```bash
rsync -a --partial --progress qudgks@210.107.192.239:workspace/clear-ec/package_20261114/upload_20261114/SHA256SUMS_p3b.txt ~/Downloads/clear-ec-upload_20261114/
```
```bash
cd ~/Downloads/clear-ec-upload_20261114 && shasum -a 256 -c SHA256SUMS_p3b.txt
```
```bash
python3 ~/Downloads/clear-ec-upload_20261114/p3b_model_patch/rebuild_model_archive_20261114.py ~/Downloads/clear-ec-upload_20261113/model_freeze-9687076-p_20261112.tar.gz ~/Downloads/clear-ec-upload_20261114/p3b_model_patch ~/Downloads/clear-ec-upload_20261114/model_freeze-9687076-p3b_20261114.tar.gz
```
- 패치 확인은 5줄 모두 OK가 나와야 합니다. 재구성의 마지막 줄은 `[5/5] CONTENTS MATCH the box-tested tree (443 files)`로 시작해야 합니다.
- Comment
  - Containers: `freeze 9687076 tier p3b 20261114 sha 67a8452f`.
  - Models: `freeze 9687076 tier p3b model sha <재구성 sha 앞 8자>`.

**p3b Try-out 기대값** (T4 경로 강제, 박스 4090; 허용 범위 CD ±0.1 %, CV ±0.5 %, HEX ±0.5 %)

| 파일 (`~/Downloads/`) | 이미지 | 기대 CD | 기대 CV | 기대 HEX |
|---|---|---:|---:|---:|
| `d4e8cdb0-765a-4b07-8091-d7da44ab6068.mha` | 0001-19 | 2854.7 | 0.3819 | 0.5462 |
| `58c72195-a5a7-48de-abea-02b3a3519a75.mha` | 0001-20 | 1999.2 | 0.4796 | 0.4134 |
| `767b441c-8a94-47e1-ad91-02f8757bcde9.mha` | 0001-21 | 4373.7 | 0.2977 | 0.6496 |

- **올린 tier의 표로 비교해 주세요.**
  - 0001-19는 p3 표와 -0.17 %, 0001-20은 -0.24 % 차이가 납니다. 둘 다 허용 범위(±0.1 %)보다 큽니다.
  - 0001-21은 두 tier 차이가 0.06 %라서 이것만으로는 구분되지 않습니다.
- `decisions`의 마지막 항목은 `"D-COORD-P3-PREP tier p3b: cd = deploy block of experiments/lane_cd_20261108/reads/freeze_cd_params_p3b_20261114.json (md5 0442dd5abcf67d93ae722374bfe7cd9a)"`입니다.
- 나머지 확인 항목은 위 4절과 같습니다(Status, 270초, recipe, errors {}, t4_default, fp16, det_copies 1).
