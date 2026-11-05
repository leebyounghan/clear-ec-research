"""Read-only experiment catalog. Never modifies training sources or artifacts.
Metric rows retain source JSON pointer; historical scores are not pooled across scopes.
"""
import argparse,concurrent.futures,datetime,hashlib,json,subprocess,math
from pathlib import Path


def metric(value):
    if not isinstance(value,dict):return None
    if all(isinstance(value.get(k),(int,float)) for k in ('CD','CV','HEX')):
        a=[value[k] for k in ('CD','CV','HEX')]
    else:
        a=next((value.get(k) for k in ('CD_CV_HEX','metrics','mape') if isinstance(value.get(k),list) and len(value[k])==3 and all(isinstance(v,(int,float)) for v in value[k])),None)
    if a is None or not all(math.isfinite(v) for v in a):return None
    return dict(zip(('CD','CV','HEX'),a),mean=value.get('mean',value.get('overall',sum(a)/3)),n=value.get('n'))


def extract(value,pointer=''):
    rows=[]
    m=metric(value)
    if m:rows.append(dict(pointer=pointer or '/',**m))
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('history','epochs_detail','epoch_artifacts','hashes','source_sha256','artifact_sha256','inputs','files'):continue
            if isinstance(v,dict):rows+=extract(v,pointer+'/'+k)
        for k in ('val','test'):
            a=value.get(k)
            if isinstance(a,list) and len(a)==3 and all(isinstance(v,(int,float)) for v in a):rows.append(dict(pointer=pointer+'/'+k,**dict(zip(('CD','CV','HEX'),a)),mean=sum(a)/3,n=None))
    return rows


def load(path):
    try:
        r=subprocess.run(['python3','-c','import sys;sys.stdout.buffer.write(open(sys.argv[1],"rb").read())',str(path)],capture_output=True,timeout=5,check=True)
        d=json.loads(r.stdout)
        return {'source':str(path),'sha256':hashlib.sha256(r.stdout).hexdigest(),'metrics':extract(d),'status':d.get('status') if isinstance(d,dict) else None}
    except Exception as e:return {'source':str(path),'read_error':type(e).__name__}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.repo=args.repo.resolve()
    paths=subprocess.check_output(['rg','--files','results'],cwd=args.repo,text=True).splitlines()
    keep=[];families={}
    for rel in paths:
        parts=Path(rel).parts
        if parts[1].startswith('research_catalog_'):continue
        family=parts[1] if len(parts)>2 else 'early_single_files'
        families[family]=families.get(family,0)+1
        if not rel.endswith('.json') or any(x in parts for x in ['observations','source','persistent_execution_20261017']):continue
        name=Path(rel).name
        if name in ['result.json','audit.json','comparison.json','screen_audit.json','screen.json','screen_summary.json','ssl_ablation.json'] or len(parts)==2 or name.endswith('_independent_audit.json'):
            keep.append(args.repo/rel)
    args.out.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:records=list(pool.map(load,keep))
    out={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Source-indexed historical metrics; not a unified leaderboard. Scores from different folds, subsets, calibration and checkpoint selection must not be subtracted.','families':families,'records':records}
    (args.out/'catalog.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>CLEAR-EC experiment catalog</title><style>body{font:14px system-ui;margin:32px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left}th{background:#eee}small{color:#555}</style><h1>CLEAR-EC 실험 원본 지표 목록</h1><p>범위가 다른 점수는 직접 비교하지 않습니다. /score는 원본 JSON 위치이며 best/terminal 여부는 원본을 확인하세요.</p><table><tr><th>실험 / 파일</th><th>JSON 위치</th><th>CD</th><th>CV</th><th>HEX</th><th>평균</th><th>n</th></tr>']
    from html import escape
    for rec in records:
        for row in rec.get('metrics',[]):
            cells=[str(Path(rec['source']).relative_to(args.repo)),row['pointer']]+[f'{row[k]:.5f}' for k in ['CD','CV','HEX','mean']]+[str(row['n'] or '—')]
            html.append('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in cells)+'</tr>')
    html.append('</table>');(args.out/'catalog.html').write_text('\n'.join(html))
    print(json.dumps({'families':len(families),'source_records':len(records),'metric_rows':sum(len(r.get('metrics',[])) for r in records),'read_errors':[r['source'] for r in records if 'read_error' in r]}))
if __name__=='__main__':main()
