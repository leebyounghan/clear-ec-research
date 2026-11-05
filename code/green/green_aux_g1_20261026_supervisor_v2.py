"""GREEN-G1 amendment: run aux_w0p01 on lane B after the original supervisor releases it.

The original supervisor skips the withdrawn aux_w0p3 (no CPU evidence) and parks
lane B under gpu_keepalive.py; this stops that keepalive, reuses the original
run_arm gates unchanged, then writes comparison_v2.json once both supervisors end.
"""
import importlib.util
import os
import pathlib
import time

spec = importlib.util.spec_from_file_location('sup1', 'experiments/green_aux_g1_20261026/supervisor.py')
sup1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup1)
R = sup1.R
ARM, GPUS = 'aux_w0p01', '2,3'
sup1.ARMS[ARM] = '0.01'


def take_lane_b():
    marker = R/'keepalive_laneB.json'
    while not marker.exists():
        time.sleep(60)
    pid = sup1.read(marker)['pid']
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        pass
    for _ in range(60):
        if not sup1.busy(GPUS):
            return
        time.sleep(10)
    raise RuntimeError('lane B still busy')


def compare_v2():
    def get(p):
        return sup1.read(p) if pathlib.Path(p).exists() else None
    control = get(R/'control_w0/aux_audit.json')
    historical = get('experiments/phase2_pooling_20261020/arms/mean/audit.json')
    out = dict(created_utc=sup1.now(), control=control, amendment='protocol_amendment_20261026.json',
               control_minus_historical_fold4_seed46_mean=(
                   {k: control['score'][k]-historical['score'][k] for k in ('CD', 'CV', 'HEX', 'mean')}
                   if control and historical else None), arms={})
    for arm in ('aux_w0p01', 'aux_w0p1'):
        a = get(R/arm/'aux_audit.json')
        d = {k: a['score'][k]-control['score'][k] for k in ('CD', 'CV', 'HEX', 'mean')} if a and control else None
        out['arms'][arm] = dict(audit=a, delta_aux_minus_control=d,
                                improves_at_least_0p03=bool(d and d['mean'] <= -0.03),
                                component_regression_over_0p05=[k for k in ('CD', 'CV', 'HEX') if d and d[k] > 0.05],
                                validation_mean_f1=a and a['validation_mean_f1'],
                                frozen_probe_validation_mean_f1=a and a['frozen_probe_validation_mean_f1'])
    out['withdrawn'] = dict(aux_w0p3=get(R/'aux_w0p3/withdrawn.json'))
    out['gate'] = 'see protocol.json and amendment; promotion and G2 are never automatic'
    sup1.save(R/'comparison_v2.json', out)


if __name__ == '__main__':
    assert not (R/'started_v2.json').exists()
    sup1.save(R/'started_v2.json', dict(started_utc=sup1.now(), pid=os.getpid(), supervisor_sha=sup1.sha(__file__)))
    try:
        take_lane_b()
        sup1.run_arm(ARM, GPUS)
    finally:
        sup1.keep_busy('B2', GPUS)
    while not (R/'exit.json').exists():
        time.sleep(60)
    compare_v2()
    sup1.save(R/'exit_v2.json', dict(finished_utc=sup1.now(), arm=sup1.read(R/ARM/'exit.json')))
