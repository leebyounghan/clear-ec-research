# -*- coding: utf-8 -*-
"""Build a scope-separated research report; never changes experiment artifacts."""
from pathlib import Path
from html import escape as e
import json,datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research_catalog_20261020'
CAT=json.loads((OUT/'catalog.json').read_text())
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><title>CLEAR-EC 연구 타임라인</title><style>body{max-width:1300px;margin:40px auto;padding:0 24px;font:15px/1.6 system-ui;color:#182333;background:#f7f9fc}h1,h2{color:#14395c}section{background:white;padding:24px;margin:22px 0;border-radius:12px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #dde4ed;text-align:left}th{background:#edf3fa}small{color:#54667a}.good{color:#08764d}.bad{color:#b44432}a{color:#1765a5}.scroll{overflow:auto}</style><h1>CLEAR-EC 실험·제출 타임라인</h1><p>2026-10-20 정리 · 낮을수록 좋은 MAPE(%) · Δ는 percentage point. 공식 제출과 내부 개발 검증을 구분합니다.</p>''']
def section(title,text=''):
 parts.append('<section><h2>'+e(title)+'</h2>'+text)
def end():parts.append('</section>')
def table(headers,rows):
 parts.append('<div class="scroll"><table><tr>'+''.join('<th>'+e(str(x))+'</th>' for x in headers)+'</tr>')
 for row in rows:parts.append('<tr>'+''.join('<td>'+e(str(x))+'</td>' for x in row)+'</tr>')
 parts.append('</table></div>')
def source(path):parts.append('<p><small>근거: <a href="'+e(str(ROOT/path))+'">'+e(path)+'</a></small></p>')
def metrics(title,rows,src):
 section(title);table(['설정','CD','CV','HEX','평균','ΔCD / ΔCV / ΔHEX / Δ평균 (첫 행 기준)'],[[name]+[f'{v:.4f}' for v in values]+['—' if i==0 else ' / '.join(f'{v-b:+.4f}' for v,b in zip(values,rows[0][1]))] for i,(name,values) in enumerate(rows)]);source(src);end()
section('실제 제출 — Phase 1', '<p><b>9월 14일 제출 8.7010은 사용자 확인과 플랫폼 연결 기록으로 확인했습니다. Phase 2 제출은 0건입니다.</b> 아래는 저장소에서 확인한 제출 목록이며 플랫폼의 전체 제출 이력을 뜻하지 않습니다.</p>')
table(['제출일 (2026)','모델','공식 점수','직전 대비','공식 CD / CV / HEX'],[['09-10','6모델 baseline + 고정 보정','8.7042','—','세부 점수 기록 미확보'],['09-14','앞 5모델 유지 + 6번째 full_quality_seed77 SSL/raw 교체 + 기존 고정 보정','8.7010','−0.0032','세부 점수 기록 미확보'],['Phase 2','내부 연구만 진행','미제출','—','—']])
source('docs/legacy_phase1/PLATFORM_BINDING_20260915.json');source('docs/legacy_phase1/STATUS_20260915.md');end()
section('날짜별 진행표', '<p>날짜는 실험 ID의 날짜 및 문서상의 진행 기간입니다. 모든 행을 정확한 실행 시작·종료시각으로 해석하면 안 됩니다. 최종 제출 패키지는 9월 14일 미국 동부시간에 완료됐으며 UTC로는 일부 작업이 9월 15일입니다.</p>')
table(['기간','단계','실험·산출물','상태 / 판단'],[
['8월–9월 초','Phase 1 초기','single/specialist, ensemble 초기 결과','과거 내부 validation/test; 후속 OOF와 분리'],
['09-02–04','기본 학습·seed·context','중앙 crop 20개 + context 8개, 3 seed','후속 후보 탐색 기반'],
['09-04–05','loss / bagging','metric35, predbag2_35, hybrid, 3→5모델','5모델 구성 선택'],
['09-05–10','region / quality / 패키징','full-quality 6번째 모델 추가, 6모델 보정 고정','09-10 공식 8.7042'],
['09-11–13','crop / target / representation','crop screen, raw targets, masked SSL, DINO','제출 후보와 연구 실험; 전부 제출한 것은 아님'],
['09-14','SSL 길이·CPSAM 방법','SSL 2000/4000/6000, 6번째 SSL/raw all-data 학습','09-14 공식 8.7010; 6번째 교체'],
['09-15','all-members SSL','다른 멤버까지 SSL 적용 실험','8.7010 제출 모델 구성과 다름; 승격 미확인'],
['09-23','ensemble 분석','멤버 제거·보정·오차 분석','재사용 개발자료의 사후 진단'],
['09-28–10-02','Phase 2 초기','metric35 SSL, fullnative, CPSAM scale, backbone screen, CPDINO foundation','아키텍처·해상도·초기화 탐색'],
['10-03–09','backbone SSL','LeJEPA 총 5/10 epoch 및 downstream regression','SSL loss와 회귀 성능을 분리 평가'],
['10-12–14','회귀 튜닝','head LR × clip 6설정 (fold3)','head1e-4/clip3 미세 우위; 독립 일반화 확정 아님'],
['10-15–19','다른 fold·seed 재검증','fold4/fold0 fresh SSL, SSL0 대조, seed43, schedule10/20 seed44·45','fold4에서 LR/clip 후보 우위 재현 실패; schedule10 유망'],
['10-20 진행 스냅샷','pooling','mean / mean_std / std 완료, min / max / 4종 결합 후속','완료 3개만 수치 표시; 최종 승자 미확정'],
['pooling 이후 예정','backbone SSL 길이','같은 CPDINO에서 SSL0/5/10/20, 동일 회귀조건','승인된 대기 계획; 완료 결과로 표기하지 않음']]);end()
section('Phase 1 최종 6모델 및 보정의 출처')
table(['멤버','seed','crop 수','loss','최종 업데이트'],[['seed42',42,1,'표준화 log L1',3576],['seed77',77,1,'표준화 log L1',2831],['seed2026',2026,1,'표준화 log L1',2980],['metric35',42,1,'log + relative mixed',4470],['predbag2_35',42,2,'표준화 log L1',3874],['full_quality_seed77 (SSL/raw)',77,2,'mixed; raw labels',5215]])
parts.append('<p>각 모델은 여러 crop·D8 view를 집계하고, 복원한 6개 예측을 동등 평균한 뒤 CD ×0.9970446111, CV ×0.9879721308, HEX ×0.9892234872를 적용합니다. 보정은 9월 7일 5-fold OOF의 예측/정답으로 MAPE를 최소화하는 가중 중앙값에서 나왔습니다. 6번째 모델을 SSL/raw로 바꿀 때 새로 맞추지 않았습니다. 보정한 동일 OOF 점수는 완전히 nested한 독립 평가가 아닙니다.</p><p>패키지 구분: 20260904 3모델 후보 → 20260905 5모델 후보 → 20260909 6모델 baseline → 20260913 raw 후보 → <b>20260914 SSL/raw 실제 제출</b>. 저장소 submission/inference.py의 오래된 4-specialist 모델을 최종 제출 구현으로 혼동하면 안 됩니다.</p>')
source('docs/legacy_phase1/METHODOLOGY.md');source('docs/legacy_phase1/SSL_RELEASE_20260914.md');end()
metrics('Phase 1 내부 OOF — 5 folds, 8,954장 (공식 점수 아님)',[
('seed42 단일',[6.792635878,10.546657745,10.28794844,9.209080688]),
('3 seed 동등 평균',[6.558504399,10.434774114,10.226306382,9.073194965]),
('5모델 동등 평균',[6.538585685,10.382685967,10.185301015,9.035524222]),
('5모델 고정 보정',[6.534406422,10.323016532,10.14903314,9.002152031])], 'docs/legacy_phase1/METHODOLOGY.md')
metrics('Phase 1 6번째 모델 비교 — matched folds2–4, 5,350장',[
('5모델',[6.623530249,10.375086537,10.156769183,9.051795323]),
('6모델 raw',[6.587403143,10.357839434,10.142312459,9.029185012]),
('6모델 SSL/raw',[6.568874749,10.358419711,10.131502979,9.019599146]),
('6모델 SSL/raw + 고정 보정',[6.565300236,10.322784929,10.089845324,8.992643496])], 'results/ensemble_ablation_20260923/report.json')
metrics('Phase 2 fold3 — full validation 1,791장',[
('CPDINO foundation',[7.299675941,10.231935501,10.373564720,9.301726341]),
('SSL5 head1e-3 clip1 terminal20',[7.058675766,10.340623856,10.321462631,9.240254402]),
('SSL5 head1e-4 clip3 terminal20',[7.009033203,10.324193954,10.364694595,9.232640266])], 'results/phase2_lejepa5_sweep_20261013/audit.json')
metrics('Phase 2 fold4 — full validation 1,790장 · terminal20',[
('seed42 SSL5 baseline',[6.5305023,10.4862061,10.3210793,9.112595558]),
('seed42 SSL5 head1e-4 clip3',[6.5631022,10.4857140,10.2970524,9.115289688]),
('seed42 SSL0 baseline',[6.6518679,10.4507332,10.3164015,9.139667511]),
('seed43 SSL5 baseline',[6.5725608,10.4828615,10.2853518,9.113591194]),
('seed43 SSL5 head1e-4 clip3',[6.6379375,10.5198107,10.3544970,9.170748711]),
('seed44 schedule20',[6.5675406,10.4294472,10.2934256,9.096804619]),
('seed44 schedule10',[6.6326933,10.3306398,10.20749,9.056941032]),
('seed45 schedule20',[6.6357360,10.4810085,10.2565699,9.124438286]),
('seed45 schedule10',[6.5230789185,10.3779993057,10.1766614914,9.0259132385])], 'results/phase2_crossfold_20261015/dispatch.json')
metrics('진행 중 pooling — fold4 1,790장 · seed46 · terminal20',[
('mean',[6.811375141,10.331394196,10.257061005,9.133276939]),
('mean + std',[6.589098454,10.380795479,10.213973045,9.061289787]),
('std',[6.660743237,10.262130737,10.220822334,9.047898293])], 'results/phase2_pooling_20261020/std_independent_audit.json')
section('해석과 후속 순서')
parts.append('<p>① 6모델 구성과 고정 보정은 내부 오차를 줄였지만 공식 개선은 8.7042→8.7010으로 작았습니다. 내부 평균과 공식 평균을 같은 시계열로 연결하지 않습니다.</p><p>② fold3의 head LR/clip 후보 우위는 fold4·새 seed에서 유지되지 않았습니다. 반면 schedule10은 seed44·45 모두 평균이 낮았습니다. 총 학습은 20epoch이며 조기 종료 10epoch와 다릅니다.</p><p>③ pooling 완료 3개 중 std는 mean보다 CD/CV/HEX 모두 낮지만, mean+std보다 CD는 높습니다. 결합 pooling은 head 입력 차원과 파라미터 수도 늘어나므로 pooling 통계만의 순수 효과로 단정하지 않습니다. 과거 seed45와 현재 seed46의 단순 순위는 matched seed 효과가 아닙니다.</p><p>④ SSL backbone이 천장이라는 근거는 아직 없습니다. pooling 비교를 끝낸 뒤 같은 mean head·fold·회귀 seed/schedule에서 SSL0/5/10/20을 비교하고, 유망 설정을 다른 fold에서 확인합니다. 기존 6모델과 비교할 때 같은 1,778장 subset을 사용하고 단일/앙상블·TTA·보정을 명시합니다.</p><p>모든 Phase 2 값은 재사용한 개발 fold 결과입니다. 환자 독립성과 near-duplicate 독립성은 미검증입니다. terminal20과 사후 best epoch는 별개이며, 공식 제출 앱을 변경하지 않았습니다.</p>');end()
section('전체 실험 파일 인덱스와 수집 범위')
table(['결과 계열','발견 파일 수'],sorted(CAT['families'].items()))
errors=[r for r in CAT['records'] if 'read_error' in r]
parts.append('<p>'+str(len(CAT['records']))+'개 원본 JSON 선택, '+str(sum(len(r.get('metrics',[])) for r in CAT['records']))+'개 지표 레코드. 동일 결과의 중복·delta·사후 진단이 포함되므로 고유 실험 수가 아닙니다. 원본 JSON 위치와 SHA256은 catalog.json에 남겼습니다.</p><p><a href="catalog.html">전체 원본 지표 표</a> · <a href="catalog.json">기계 판독 catalog JSON</a></p>')
if errors:table(['읽기 미완료 원본 (수치 누락을 0으로 처리하지 않음)','원인'],[[r['source'],r['read_error']] for r in errors])
parts.append('<p>리팩토링 범위: code/reporting 아래로 결과 수집·보고서 생성을 분리했습니다. 기존 학습 소스, frozen manifest, 체크포인트 및 실행 중 프로세스는 수정하지 않습니다. 원본 schema마다 다른 범위를 임의로 하나의 leaderboard로 합치지 않습니다.</p>');end()
parts.append('</html>');(OUT/'timeline.html').write_text('\n'.join(parts))
print(OUT/'timeline.html')
