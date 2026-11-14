# CLEAR-EC 외부 데이터 조사 보고 (2026-11-06)

## 결론

- **새로 받은 공개 데이터에는 쓸 만한 학습 이미지가 없습니다.** 공개된 스펙큘러 이미지는 이미 가진 SREP 385장이 사실상 전부였습니다.
- **받은 것 중 시험해 볼 가치가 가장 큰 것은 Rotterdam 가중치입니다.** 1203장 논문용 가중치라, 신청 없이 그 학습 결과를 일부 활용할 수 있습니다. 아직 돌려 보지 않아 효과는 모릅니다.
- **큰 데이터는 전부 신청이 필요하고, 이번 제출에 맞을지는 불확실합니다.** 오늘이 금요일이고 동결은 11-11 저녁이라, 답이 늦으면 학습할 시간이 없습니다.
- **먼저 할 일은 CPTS 웹 양식 제출입니다.** 기증 각막 이미지가 포함되어 도메인이 가장 가깝고, PI가 몇 분이면 제출할 수 있습니다.
- **챌린지 규칙은 미확인입니다.** 로그인 없이 읽을 수 있는 페이지에는 외부 데이터 규칙이 없습니다.

## 챌린지 규칙 (모든 항목 공통)

- 메인 페이지, 세부 안내 페이지(opticlab), 베이스라인 README, Zenodo 라이선스 요약을 읽었고, 외부 데이터나 사전학습 모델을 허용하거나 금지하는 문구는 없었습니다.
- 베이스라인 자체는 사전학습된 Cellpose 가중치를 씁니다.
- "사람 주석 금지"도 이 페이지들에는 없습니다. 현재 출처는 BRIEF이고, 로그인 페이지나 포럼에 있을 수 있습니다.
- 로그인이 필요한 페이지와 포럼은 읽지 못했습니다. 확정하려면 주최 측에 물어야 합니다(초안 5번).

## 오늘 받은 것

위치: `/workspace/clear-ec-research/datasets/external_20261106/` (파드 `clear-ec-research-4gpu-1104`), 합계 약 703 MB. 폴더마다 출처, 라이선스, 체크섬이 적힌 README가 있고, 체크섬 재검증을 통과했습니다. 받은 코드와 가중치는 실행하거나 로드하지 않았습니다.

| 폴더 | 내용 | 이미지 | 라벨 | 라이선스 | 더할 수 있는 것 |
|---|---|---:|---|---|---|
| `vigueras_fnla_weights` | 1203장 논문의 가중치 22개 | 0 | 신경망 가중치 | GPL-3.0 | 챌린지 라벨 없이 돌릴 수 있는 별도 경계 탐지기. **미검증** |
| `vigueras_tvst_weights` | 383장 논문의 가중치 | 예시 1장 | 신경망 가중치 | GPL-3.0 | 위와 같음, 이식 후 각막용 |
| `brookes_nz_voronoi_table` | 뉴질랜드 안은행 세포별 측정표 | 0 (2000장 분량) | 세포 354,998개의 면적, 변 수 | CC BY 4.0 | 시뮬레이션 테셀레이션 분포 보정 |
| `osf_prada_fuchs_tables` | 푸크스 환자 측정표 | 0 (1352행) | AI CD, CV, HEX, 기기 밀도 | CC BY 4.0 | 낮음. 기증 각막이 아니고 환자 이니셜 열이 있음 |
| `kucharski_gan_alizarine` | Alizarine 사본, GAN 생성기 | 돼지 30장 | 경계, 마커, ROI | 비상업. Alizarine 조건 미확인으로 **격리** | 낮음. 염색 조직이고 스펙큘러가 아님 |
| `gavet_focus_stacks` | 체외 초점 스택 2개 | 75프레임 | 없음 | "free to use", 인용 요구 | 낮음. 초점 흐림 증강의 참고 자료 (가설) |
| `sierra_sdm_guttae` | 코드, 학습된 모델 | 예시 1장 | 가중치, 분할 1장 | Zenodo "other-open" | 낮음. 압축 해제 안 함 |

측정표 값 (Brookes, 이미지별 중앙값): 밀도 2929, 면적 CV 17.5 %, 육각 비율 0.511, 이미지당 세포 182개. BRIEF의 "기기 CV는 Voronoi CV의 약 2배"와 어긋나지 않는 수준이지만, 모집단이 달라 검증은 아닙니다.

## 신청이 필요한 것 (초안 작성, 미발송)

초안: [external_data_requests_20261106.md](external_data_requests_20261106.md)

| 데이터셋 | 이미지 | 라벨 | 라이선스 | 상태 | 더할 수 있는 것 |
|---|---:|---|---|---|---|
| CPTS 공개 이미지 세트 (Case Western) | 수량 미기재. 프로토콜 목표는 1330안, 수술 전 기증 각막 3장씩과 수술 후 추적 영상 | 미기재 | 미기재 | 웹 양식, PI 제출 필요 | 기증 각막 도메인 사전학습. **가장 가까움** |
| Rotterdam Topcon SP-1P | 1203 (500장에 guttae) | 전 이미지 수동 분할, 기기 CD, CV, HEX | 요청 시 협의 | 저자 메일 초안 | 탐지기 학습, 기기값과 수동값 대조 |
| ROD-Rep Topcon SP-1P | 50 | 수동 분할, Topcon 분할 | 논문은 무료 공개라고 기재 | 사이트 다운, 저자 메일에 포함 | C2 기기 공식 감사에 직접 쓸 수 있음 |
| Case Western 기증 각막 (Konan CellChek D) | 라벨 174, ECD 있는 15,138 | 수정된 경계 분할, 안은행 ECD | 공개 문구 없음 | 저자 메일 초안 | 챌린지와 해상도가 같음(1296 x 972). 다만 공유 가능성은 낮아 보임 |
| CED (Zenodo 23540380) | 1200 | 세포 단위 주석, 질환 라벨 | CC BY 4.0, 파일은 제한 | Zenodo 로그인 후 신청 | 기기와 해상도 미기재라 판단 불가 |
| TM-EM3000 (Tomey) | 184 | 분할 | 없음 | 짧은 초안 | 다른 스펙큘러 기기 |
| Sierra guttae 세트 | 90 | 분할 | 없음 | 짧은 초안 | 소량, 건강안과 푸크스 환자 |
| MCCM (Manchester) | 1010 | 일부 수동 경계 | 없음 | 짧은 초안, 주소 못 찾음 | 공초점이라 가치 낮음 |

## 이미 가진 것 (다시 받지 않음)

위치: `datasets/external/`. SREP Topcon SP-3000 385장(세포 중심점), RODREP 공초점 52장, GitHub U-Net 160장(격리), FECD 평판 499장.

## 받지 않은 것

| 항목 | 이유 |
|---|---|
| Indhira2025 GitHub 리포 | SREP 사본. 파일 770개의 이름과 크기가 전부 일치 |
| jyothi6799 GitHub 리포 | 격리본의 사본. 파일 320개가 전부 격리본에 있음 |
| sgrzegorz GitHub 리포 (약 119세트, 152 x 388) | 라이선스와 출처가 없어 보류 |
| Gavet 30장 참조 세트 | 다운로드 위치를 찾지 못함. 생성기만 받음 |
| BioImLab 원본 사이트 | DNS가 응답하지 않음 |
| Kaggle | 검색 결과 없음 |
| Hugging Face `davidm770/corneal` | 설명 없는 h5 파일 1개. 내용을 알 수 없어 받지 않음 |
| Roboflow | "Corneal" 140장 분할 세트 1건이 있으나 내용은 확인하지 못함 |

## 알아 두실 점

- **BRIEF 수정 필요:** 1203장과 383장은 별개 세트가 아닙니다. 1203 = Baerveldt 연구 820 + UT-DSAEK 383입니다. Baerveldt 연구의 원본은 7975장입니다. BRIEF는 고치지 않았습니다.
- **기기값 주의:** Rotterdam 논문은 Topcon 내장 소프트웨어의 오차가 자기 방법의 3~6배라고 보고합니다. 기기 CV, HEX를 라벨로 쓰기보다 수동 분할에서 다시 계산하는 편이 안전합니다.
- **해상도 차이:** Rotterdam 이미지는 약 1.04 um/px, 챌린지는 0.7716 um/px입니다. 가중치를 쓰려면 배율을 맞춰야 합니다.
- **가중치 출처:** 리포 README는 가중치가 어떤 데이터와 fold로 학습됐는지 밝히지 않습니다. 1203장 학습은 논문 내용에서 추정한 것입니다.
- **GPL-3.0:** 가중치를 제출 컨테이너에 넣기 전에 라이선스를 확인해야 합니다. 가중치에 대한 별도 조항은 없습니다.
- **이전 방침과 다른 점:** 예전 인벤토리는 Alizarine을 재게시본에서 받지 않기로 했습니다. 이번에는 "공개된 것은 받는다"는 지시에 따라 논문 저자 리포에서 받았고, 격리 표시를 했습니다. 원치 않으시면 폴더를 지우면 됩니다.
- **중복 확인 필요:** CPTS 이미지는 받으면 챌린지 학습 이미지와 대조해야 합니다. 참여 안은행 명단을 찾지 못해 겹침 여부를 모릅니다.

## 출처

- Vigueras-Guillén et al., Scientific Reports 2022: https://doi.org/10.1038/s41598-022-18180-1
- Vigueras-Guillén et al., TVST 2020: https://doi.org/10.1167/tvst.9.2.49
- Vigueras-Guillén et al., BMC Biomedical Engineering 2019: https://doi.org/10.1186/s42490-019-0003-2
- Benetz et al., TVST 2024: https://doi.org/10.1167/tvst.13.8.40
- CPTS 이미지 세트: https://case.edu/medicine/ophthalmology/research/clinical-research/cornea-preservation-time-study-cpts-public-specular-microscopy-image-set
- CPTS 프로토콜: https://cdn.clinicaltrials.gov/large-docs/93/NCT01537393/Prot_SAP_000.pdf
- Brookes 측정표: https://doi.org/10.17608/k6.auckland.5701087
- Prada et al. 데이터: https://osf.io/75kmu/
- CED: https://zenodo.org/records/23540380
- 챌린지: https://clear-ec.grand-challenge.org/ , https://opticlab.med.upenn.edu/clear-ec-challenge
