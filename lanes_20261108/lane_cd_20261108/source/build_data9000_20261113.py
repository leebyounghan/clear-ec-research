#!/usr/bin/env python3
"""A-CD-DATA9000 (coordinator 23:40 EST 11-13): new dataset folder datasets/clear_ec_fullframe9000_20261113/ (nothing existing modified):
img_full_raw9000.dat = the 8,954 frames of img_full_raw.dat in the same order + the 46 frames lost to the ID spelling match, read from train_mha/
(raw_id_alias_audit.json: exact matches first, whitespace/hyphen removal only for unmatched IDs). meta9000.npz (ids = file-name IDs, csv_ids,
Y from final_train_ids.csv, N, SZ), split_full9000.npz (train = every row with three positive labels = 8,998; validation = the two HEX-zero rows),
manifest.json with the checks. Run from /workspace/clear-ec-research, one niced process."""
import os, json, hashlib, numpy as np, pandas as pd, SimpleITK as sitk
D = 'datasets/clear_ec_fullframe9000_20261113'; os.makedirs(D, exist_ok=False)
SRC = 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'; FR = 972 * 1296; N0 = 8954
meta = np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True); ids0 = meta['ids'].astype(str); Y0 = meta['Y'].astype(np.float32)
assert len(ids0) == N0 and os.path.getsize(SRC) == N0 * FR
csv = pd.read_csv('datasets/clear_ec_official/final_train_ids.csv', dtype=str, keep_default_na=False)
al = json.load(open('datasets/clear_ec_fullframe_20260904/raw_id_alias_audit.json'))['aliases']; assert len(al) == 46
out = {}
# labels of the 8,954 equal the CSV rows (exact ID match)
cmap = {r.ID: r for r in csv.itertuples()}
Yc = np.array([[float(cmap[i].CD), float(cmap[i].CV), float(cmap[i].HEX)] for i in ids0], np.float32)
out['labels_8954_equal_csv_max_abs'] = float(np.max(np.abs(Yc - Y0)))
# copy with chunked sha256
hs, hd = hashlib.sha256(), hashlib.sha256(); dst = f'{D}/img_full_raw9000.dat'; CH = 64 * FR
with open(SRC, 'rb') as fi, open(dst, 'wb') as fo:
    while True:
        c = fi.read(CH)
        if not c: break
        hs.update(c); fo.write(c)
    ids_new, csv_new, Y_new, frame_sha = [], [], [], []
    for a in al:
        img = sitk.GetArrayFromImage(sitk.ReadImage(f"datasets/clear_ec_fullframe_20260904/train_mha/{a['raw_id']}.mha"))
        img = np.squeeze(img); assert img.shape == (972, 1296) and img.dtype == np.uint8, (a['raw_id'], img.shape, img.dtype)
        r = cmap[a['source_csv_id']]; fo.write(np.ascontiguousarray(img).tobytes())
        ids_new.append(a['raw_id']); csv_new.append(a['source_csv_id']); Y_new.append([float(r.CD), float(r.CV), float(r.HEX)]); frame_sha.append(hashlib.sha256(img.tobytes()).hexdigest())
with open(dst, 'rb') as f:
    left = N0 * FR
    while left:
        c = f.read(min(CH, left)); hd.update(c); left -= len(c)
out['first_8954_sha256_source'] = hs.hexdigest(); out['first_8954_sha256_new'] = hd.hexdigest(); out['first_8954_identical'] = hs.hexdigest() == hd.hexdigest()
out['new_file_bytes'] = os.path.getsize(dst); out['expected_bytes'] = 9000 * FR
ids = np.array(list(ids0) + ids_new); csv_ids = np.array(list(ids0) + csv_new); Y = np.concatenate([Y0, np.array(Y_new, np.float32)])
out['unique_ids'] = int(len(set(ids))); out['rows'] = int(len(ids)); out['46_all_positive_labels'] = bool((np.array(Y_new) > 0).all())
out['46_frame_sha_unique'] = len(set(frame_sha)) == 46
np.savez(f'{D}/meta9000.npz', ids=ids, csv_ids=csv_ids, Y=Y, N=np.int64(9000), SZ=meta['SZ'])
tr = np.where(np.isfinite(Y).all(1) & (Y > 0).all(1))[0].astype(np.int64); va = np.setdiff1d(np.arange(9000), tr).astype(np.int64)
np.savez(f'{D}/split_full9000.npz', train=tr, validation=va); out['split_train_rows'] = int(len(tr)); out['split_held_rows'] = [str(i) for i in ids[va]]
# spot check: memmap read of an appended frame equals the .mha
mm = np.memmap(dst, mode='r', dtype=np.uint8, shape=(9000, 972, 1296)); a0 = al[0]
out['appended_frame_readback_equal'] = bool((np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(f"datasets/clear_ec_fullframe_20260904/train_mha/{a0['raw_id']}.mha"))) == mm[N0]).all())
out['rule'] = 'raw_id_alias_audit.json: exact matches first; whitespace/hyphen removal only for unmatched IDs; no fuzzy matching'
json.dump(out, open(f'{D}/manifest.json', 'w'), indent=1); print(json.dumps(out, indent=1))
