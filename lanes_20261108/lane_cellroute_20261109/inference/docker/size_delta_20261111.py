"""Sizes (os.stat, no du) of the member groups changed by the 11-11 freezes, and the current deploy manifest total. (cell-route lane)"""
import os, glob, json
os.chdir('/workspace/clear-ec-research'); E = 'experiments/'
G = {'D second seed (CD lane s47, e16-20 TTA4) [added]': [E + f'lane_cd_20261108/arms/dinov3_b16_f99_s47/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)],
     'cvw3 second seed (HEX lane s47, e08-12) [added]': [E + f'lane_hex_20261108/deploy/cvw3_f99_s47/runs/fold99/dinov3_b16/epoch_{e:02d}.pt' for e in range(8, 13)],
     'order-DINOv3 (CV, e16-20) [removed]': [E + f'lane_cv_20261108/arms/dinov3_order_f99_s46/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)],
     'HEX graph net hex_pre46_s5 [removed]': sorted(glob.glob(E + 'lane_hex_20261108/deploy/graph_models/full_hex_pre46_s5_s*_e*.pt'))}
out = {k: dict(files=len(v), gb=round(sum(os.stat(p).st_size for p in v) / 1e9, 3), missing=[p for p in v if not os.path.exists(p)]) for k, v in G.items()}
print(json.dumps(out, indent=1))
st = os.statvfs('/workspace/clear-ec-research'); print('free on /workspace: %.0f GB' % (st.f_bavail * st.f_frsize / 1e9))
