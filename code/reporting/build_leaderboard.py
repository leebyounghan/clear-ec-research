# -*- coding: utf-8 -*-
"""Offline leaderboard: scope-separated ranking, versioned notes and embedded evidence."""
import json,hashlib,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/leaderboard';OUT.mkdir(exist_ok=True)
CAT=json.loads((ROOT/'results/research_catalog_20261020/catalog.json').read_text())
rows=[]
def read(path):
 try:return subprocess.run(['cat',str(ROOT/path)],capture_output=True,timeout=3,check=True).stdout.decode('utf8')
 except Exception:return None

def add(name,group,score,source,idea,problem,decision,config='',status='결과 기록',log=None):
 ident='exp-'+hashlib.sha256((group+name).encode()).hexdigest()[:12]
 raw=read(source) if source else None
 row=dict(id=ident,name=name,group=group,score=score,source=source,config=config,status=status,problem=problem,idea=idea,decision=decision,parent=None,protocol=None,log=log,raw=raw,log_excerpt=None,note_version='v1',interpretation='2026-10-20 사후 정리. 당시 사전 가설과 구분.',date='2026-10-20',sha256=hashlib.sha256(raw.encode()).hexdigest() if raw else None)
 rows.append(row);return row
for rec in CAT['records']:
 path=Path(rec['source']);path=path if path.is_absolute() else ROOT/path
 rel=str(path.relative_to(ROOT))
 if not rel.endswith('/result.json') or '/runs/' not in rel:continue
 scores=[x for x in rec.get('metrics',[]) if x['pointer']=='/score']
 if not scores:continue
 score={k:scores[0][k] for k in ['CD','CV','HEX','mean']}
 if '/fold4/' in rel:
  branch=rel.split('/fold4/')[1].split('/runs/')[0];name=branch.replace('/arms/',' · ').replace('arms/','').replace('/',' · ')
  idea='head LR와 clipping 변경을 같은 fold에서 비교한다.';problem='fold3 후보의 작은 평균 우위가 다른 개발 fold에서도 유지되는지 불확실했다.';decision='동일 seed baseline과 비교한다. fold4 전체 순위는 서로 다른 seed를 포함한다.'
  if 'schedule' in branch:idea='총 20epoch는 유지하고 cosine schedule 길이만 10/20으로 비교한다.';problem='후기 CD/CV 개선과 HEX 악화가 함께 나타나 총 epoch를 일괄 단축할 근거가 부족했다.';decision='seed44·45 모두 schedule10 평균 개선. seed44에서는 CD 손해, seed45에서는 세 지표 모두 개선.'
  if 'ssl0_control' in branch:idea='추가 SSL 없이 외부 CPDINO 초기화로 matched 회귀 대조군을 만든다.';problem='head 튜닝 효과와 SSL encoder 효과를 구분할 필요가 있었다.';decision='seed42 SSL5 baseline과 비교. 작은 평균 차이만으로 일반화를 단정하지 않는다.'
  r=add(name,'fold4',score,rel,idea,problem,decision,'full validation1790 · 단일 모델 · terminal20 · TTA/보정 없음')
  r['protocol']='results/phase2_crossfold_20261015/research_protocol.json';r['date']='2026-10-15–19'
  if 'seed44' in branch or 'seed45' in branch:r['log']='results/phase2_crossfold_20261015/persistent_execution_20261017/'+('seed44' if 'seed44' in branch else 'seed45')+'_schedule_attempt/full.log'
  elif 'seed43' in branch:r['log']='results/phase2_crossfold_20261015/persistent_execution_20261017/seed43_attempt/full.log'
 elif '/fold3/' in rel:
  family=Path(rel).parts[1];arm=rel.split('/arms/')[1].split('/runs/')[0] if '/arms/' in rel else ''
  r=add(family.replace('phase2_','')+(' · '+arm if arm else ''),'fold3',score,rel,'CPDINO 초기화의 추가 SSL 또는 회귀 최적화 설정을 비교한다.','encoder 적응과 downstream 학습 설정 중 어떤 변경이 회귀에 유효한지 분리할 필요가 있었다.','CD/CV/HEX의 상충과 다른 fold 재현성을 함께 확인한다. SSL epoch 수 증가가 일관된 개선을 뜻하지 않는다.','full validation1791 · 단일 모델 · result.json 최종 score')
  r['date']=family[-8:];r['protocol']=None
# Sweep arm whose individual result was not locally indexed.
sweep=next((r for r in CAT['records'] if r['source'].endswith('phase2_lejepa5_sweep_20261013/audit.json')),None)
if sweep:
 for x in sweep.get('metrics',[]):
  if x['pointer']=='/scores/head3e4_clip1':
   add('lejepa5_sweep_20261013 · head3e4_clip1','fold3',{k:x[k] for k in ['CD','CV','HEX','mean']},'results/phase2_lejepa5_sweep_20261013/audit.json','head LR를 3e-4로 낮추고 clip1을 유지한다.','head LR와 clipping의 민감도를 확인한다.','baseline보다 평균이 높아 우위 근거가 없다.','validation1791 · terminal20')
poolroot=ROOT/'results/phase2_pooling_20261020'
logs=sorted(poolroot.glob('observations/*/full.log'));poollog=str(logs[-1].relative_to(ROOT)) if logs else None
latest_status=json.loads((poolroot/'latest_status.json').read_text())
latest_snapshot=logs[-1].parent.name if logs else 'unknown'
for mode in ['mean','mean_std','std','min','max','mean_std_min_max']:
 audit=poolroot/(mode+'_independent_audit.json');d=json.loads(audit.read_text()) if audit.exists() else None
 progress=latest_status.get('arms',{}).get(mode,{}).get('progress',{})
 decision='Terminal20 audited; compare against matched mean baseline. Final selection awaits all six arms.' if d else 'Saved snapshot '+latest_snapshot+': '+progress.get('status','not started')+'; epoch '+str(progress.get('epoch','-'))+'. Not a live cluster view.'
 r=add('pooling · '+mode,'fold4',d['terminal20_score'] if d else None,str(audit.relative_to(ROOT)) if d else 'results/phase2_pooling_20261020/latest_status.json','Compare spatial '+mode+' aggregation.','Test whether spatial dispersion and extrema improve CV/HEX.',decision,'seed46 / fold4 n1790 / SSL5 / epochs20 schedule10 / head1e-3 clip1','Independent audit PASS' if d else progress.get('status','not started'),poollog)
 manifests=sorted(poolroot.glob('observations/*/arms/'+mode+'/manifest.json'))
 r['protocol']=str(manifests[-1].relative_to(ROOT)) if manifests else None
 r['date']='2026-10-20';r['precision']='독립 float64 prediction 재계산' if d else None
 if 'mean_std' in mode:r['decision']+=' 통계 결합으로 head 차원·파라미터 수가 증가한다.'
for date,name,value,src in [('2026-09-10','6-model baseline + fixed calibration',8.7042,'docs/legacy_phase1/STATUS_20260915.md'),('2026-09-14','6-model SSL/raw + fixed calibration',8.7010,'docs/legacy_phase1/PLATFORM_BINDING_20260915.json')]:
 r=add(name,'official',{'mean':value},src,'6번째 full-quality 모델을 SSL/raw로 교체; 기존 5모델과 보정은 유지.' if date.endswith('14') else '6개 모델 및 고정 보정을 제출 패키지로 구성.','서로 다른 모델의 오차를 평균하고 고정적인 예측 편향을 줄이는 접근.','공식 8.7042→8.7010. 개별 CD/CV/HEX 공식 점수는 확보되지 않음.','Phase1 · 6모델 · multi-view · 고정 보정','실제 제출 확인');r['date']=date;r['protocol']='docs/legacy_phase1/SSL_RELEASE_20260914.md' if date.endswith('14') else 'docs/legacy_phase1/METHODOLOGY.md'
for name,vs in [('5-model',[6.623530249,10.375086537,10.156769183,9.051795323]),('6-model raw',[6.587403143,10.357839434,10.142312459,9.029185012]),('6-model SSL/raw',[6.568874749,10.358419711,10.131502979,9.019599146]),('6-model SSL/raw + fixed calibration',[6.565300236,10.322784929,10.089845324,8.992643496])]:
 add(name,'phase1',dict(zip(['CD','CV','HEX','mean'],vs)),'results/ensemble_ablation_20260923/report.json','동일 이미지에서 6번째 모델 및 기존 보정의 기여를 분리한다.','구성 변경의 효과와 평가 이미지 변경의 효과를 혼동하지 않도록 matched 비교가 필요했다.','5,350장 개발자료의 내부 분석. 공식 점수나 독립 test 성능으로 해석하지 않는다.','folds2–4 · matched5350 · ensemble · multi-view')
# Explicit matched baselines only, never arbitrary previous-ranked rows.
for r in rows:
 candidates=[]
 if r['name'].startswith('pooling · ') and r['name']!='pooling · mean':candidates=[x for x in rows if x['name']=='pooling · mean']
 elif 'schedule10' in r['name']:candidates=[x for x in rows if x['name']==r['name'].replace('schedule10','schedule20')]
 elif 'head1e4_clip3' in r['name'] and r['group']=='fold4':candidates=[x for x in rows if x['name']==r['name'].replace('head1e4_clip3','head1e3_clip1')]
 if candidates:r['parent']=candidates[0]['id']
for r in rows:
 if r['log']:
  raw=read(r['log']);r['log_excerpt']=raw[-16000:] if raw else None
 r['note_path']='results/leaderboard/notes/'+r['id']+'.v1.json'
 note={k:v for k,v in r.items() if k not in ['raw','log_excerpt']}
 p=ROOT/r['note_path']
 # Never rewrite an existing version with changed interpretation.
 if p.exists() and json.loads(p.read_text())!=note:
  i=2
  while (ROOT/('results/leaderboard/notes/'+r['id']+'.v'+str(i)+'.json')).exists():i+=1
  r['note_version']='v'+str(i);r['note_path']='results/leaderboard/notes/'+r['id']+'.v'+str(i)+'.json';note={k:v for k,v in r.items() if k not in ['raw','log_excerpt']};p=ROOT/r['note_path']
 p.write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n')
(OUT/'experiment_log_template.json').write_text(json.dumps({'schema_version':1,'id':'stable-experiment-id','version':'v1','written_at_utc':None,'hypothesis_written_before_run':None,'previous_experiment_id':None,'previous_limitation':'이전 결과에서 해결되지 않은 문제; 실패와 불확실성을 구분','idea':'이번 가설 / 한 가지 변경','controls':'fold, seed, encoder, LR, schedule, epoch, targets, pooling/head dimensions','evaluation_contract':'subset/count, terminal or best, single/ensemble, TTA/calibration','source_manifest_path':None,'source_manifest_sha256':None,'idea_file_at_this_version':None,'raw_log_path':None,'result_path':None,'audit_path':None,'metrics':{'CD':None,'CV':None,'HEX':None,'mean':None},'matched_baseline_id':None,'decision':'관측 사실 → 해석 → 유지/기각/재검증','limitations':[],'next_experiment':None},ensure_ascii=False,indent=2))
payload={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence_snapshot':'catalog + independent pooling audits; cluster snapshot '+latest_snapshot,'root':str(ROOT),'rows':rows}
(OUT/'experiments.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
page=(ROOT/'code/reporting/leaderboard_template.html').read_text().replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
(OUT/'index.html').write_text(page)
print(json.dumps({'experiments':len(rows),'ranked':sum(bool(r['score']) for r in rows),'path':str(OUT/'index.html')}))
