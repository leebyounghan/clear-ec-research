"""Local orchestration only: generate bounded Kubernetes Jobs for approved stages.

Never starts a Job itself. Pass its output to kubectl create; do not apply/restart.
"""
import argparse
import json

SCRIPT = '/workspace/clear-ec-research/experiments/cpsam_methods_20260914_v2/source/run_cpsam_methods_20260914.py'
PYTHON = '/workspace/venv/bin/python'


def job(name, command, seconds, stage):
    return dict(apiVersion='batch/v1', kind='Job',
        metadata=dict(name=name, namespace='p-ajou-mlxp-project', labels={'app':'clear-ec','wave':'cpsam-methods-0914-v2','stage':stage}),
        spec=dict(backoffLimit=0, activeDeadlineSeconds=seconds+600, ttlSecondsAfterFinished=600,
            template=dict(metadata=dict(labels={'app':'clear-ec','wave':'cpsam-methods-0914-v2','stage':stage}),
                spec=dict(automountServiceAccountToken=False, priorityClassName='private-zone', restartPolicy='Never',
                    nodeSelector={'mlx.navercorp.com/gpu-type':'H200','mlx.navercorp.com/hardware-type':'gpu',
                                  'mlx.navercorp.com/zone':'private-h200-ajou-mlxp-0'},
                    tolerations=[dict(key='nvidia.com/gpu',operator='Exists',effect='NoSchedule')],
                    containers=[dict(name='researcher',image='leebyounghan/mlx-rl:20260810-sgl0512-verl-1ddf071',
                        workingDir='/workspace/clear-ec-research',
                        command=['timeout','--signal=TERM','--kill-after=30s',str(seconds),PYTHON,'-u',SCRIPT]+command,
                        env=[dict(name=k,value=v) for k,v in {'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
                            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}.items()],
                        resources=dict(requests={'cpu':'6','memory':'24Gi','nvidia.com/gpu':'1'},
                                       limits={'cpu':'6','memory':'64Gi','nvidia.com/gpu':'1'}),
                        volumeMounts=[dict(name='workspace',mountPath='/workspace'),dict(name='dshm',mountPath='/dev/shm')])],
                    volumes=[dict(name='workspace',persistentVolumeClaim=dict(claimName='clear-ec-workspace')),
                             dict(name='dshm',emptyDir=dict(medium='Memory',sizeLimit='8Gi'))]))))


def manifests(stage, method=None):
    if stage=='preflight':
        return [job('ce-cpsam-gpu-preflight-0914-v2',['gpu-preflight'],1800,'preflight')]
    if stage=='ssl':
        return [job(f'ce-cpsam-{m}-{r}-0914-v2',['ssl','--tag',m+'_'+r],28800 if m=='dino' else 18000,'ssl')
            for m in ('dino','ijepa') for r in ('low','high')]
    if stage=='downstream':
        return [job(f'ce-cpsam-{m}-f{f}-0914-v2',['downstream-run','--method',m,'--fold',str(f)],25200,'downstream')
            for m in ('dino','ijepa') for f in (2,3)]
    assert method in ('dino','ijepa')
    return [job(f'ce-cpsam-{method}-candidate-0914-v2',['final-fit','--method',method],25200,'candidate')]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','ssl','downstream','candidate'))
    p.add_argument('--method',choices=('dino','ijepa'));a=p.parse_args()
    print(json.dumps(dict(apiVersion='v1',kind='List',items=manifests(a.stage,a.method)),indent=2))
