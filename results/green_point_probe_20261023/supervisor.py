import pathlib,subprocess,os,fcntl,json,datetime
r=pathlib.Path('experiments/green_point_probe_20261023')
locks=[]
for path in [r/'execution.lock',pathlib.Path('experiments/phase2_backbone_ssl_20261020/gpu0.lock')]:
 f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert not (r/'launch.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH=str(r/'source')+':/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime')
py='/workspace/venv/bin/python'
def save(n,d):
 with (r/n).open('x') as f:json.dump(d,f,indent=2)
save('launch.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=0,deadline=None))
rc=0
with (r/'full.log').open('x') as log:
 for source in ['green_point_spatial_20261023.py','green_point_probe_head_20261023.py']:
  rc=subprocess.call([py,str(r/'source'/source)],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
  if rc:break
 if rc==0:rc=subprocess.call([py,str(r/'source/green_point_run_20261023.py'),'smoke','--root',str(r)],env=env,stdout=log,stderr=subprocess.STDOUT)
 if rc==0:rc=subprocess.call([py,str(r/'source/green_point_run_20261023.py'),'train','--root',str(r)],env=env,stdout=log,stderr=subprocess.STDOUT)
save('exit.json',dict(exit_code=rc,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
