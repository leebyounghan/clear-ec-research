import sys, os, numpy as np
sys.path.insert(0, "experiments/eval_harness_20261031/source")
import eval_oof_20261031 as E
from PIL import Image
b = E.load_base(); y = b["y"]; ids = list(b["ids"])
mm = np.memmap("datasets/clear_ec_fullframe_20260904/img_full_raw.dat", dtype=np.uint8, mode="r", shape=(8954, 972, 1296))
rows = [ids.index(w) for w in ["2020-0651 OSCN", "2020-0644 OSCN", "0933-24", "2019-0064-OSCN"]]
yr = np.array([E.year_of(i) for i in ids])
ok = (yr == 2020) & (np.abs(b["B"][:, 0] - y[:, 0]) < 50) & (np.abs(y[:, 0] - 2700) < 100)
rows += list(np.where(ok)[0][:2])
rows += list(np.where((y[:, 0] > 0) & (y[:, 0] < 1500) & (np.abs(b["B"][:, 0] - y[:, 0]) / y[:, 0] < 0.15))[0][:2])
os.makedirs("/tmp/cdtail", exist_ok=True)
for r in rows:
    a = np.asarray(mm[b["idx"][r]])
    n = "/tmp/cdtail/%s_y%d_B%d.png" % (ids[r].replace(" ", "_"), int(y[r, 0]), int(b["B"][r, 0]))
    Image.fromarray(a).resize((648, 486)).save(n); print(n)
