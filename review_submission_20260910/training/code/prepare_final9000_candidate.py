"""Separate all-9000 central cache; never alter the existing OOF dataset."""
import csv
import fcntl
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from final_candidate_core import central512,save_json

RAW=Path('datasets/clear_ec_fullframe_20260904')
OUT=Path('datasets/clear_ec_final9000_20260904')
REF=Path('datasets/clear_ec_official')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'manifest.json').exists():
            print('already prepared',flush=True); return
        source=RAW/'source_metadata/final_train_ids_9000_reconciled.csv'
        with source.open(newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==9000 and len({r['ID'] for r in rows})==9000
        rawmanifest=json.loads((RAW/'fullframe_manifest.json').read_text())
        alias=json.loads((RAW/'raw_id_alias_audit.json').read_text())
        assert alias['archive_sha256']==rawmanifest['archive_sha256'] and alias['normalized_bijection']
        meta=np.load(REF/'cache_native_meta_512.npz',allow_pickle=True)
        reference={str(v):i for i,v in enumerate(meta['ids'])}
        old=np.memmap(REF/'img_native_512.dat',dtype='uint8',mode='r',shape=(len(reference),512,512))
        out=OUT/'img_native_512.dat'
        if out.exists(): raise FileExistsError('Unmanifested cache: inspect before recovery')
        pixels=np.memmap(OUT/'img_native_512.partial.dat',dtype='uint8',mode='w+',shape=(9000,512,512))
        y=np.asarray([[float(r[k]) for k in ['CD','CV','HEX']] for r in rows],dtype=np.float32)
        assert np.isfinite(y).all()
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        matched=0
        for i,r in enumerate(rows):
            pixels[i]=central512(sitk.GetArrayFromImage(sitk.ReadImage(str(RAW/'train_mha'/(r['ID']+'.mha')))))
            if r['ID'] in reference:
                j=reference[r['ID']]; assert np.array_equal(pixels[i],old[j])
                np.testing.assert_allclose(y[i],meta['Y'][j],rtol=1e-5); matched+=1
            if (i+1)%500==0: print(f'prepared {i+1}/9000',flush=True)
        assert matched==8954
        pixels.flush(); del pixels,old
        valid=(y>0).all(1)
        smoothed=y.copy(); groups=defaultdict(list)
        for i,r in enumerate(rows):
            if valid[i]: groups[r['slide_id']].append(i)
        for ix in groups.values():
            if len(ix)>=2: smoothed[ix]=y[ix].mean(0)
        np.savez_compressed(OUT/'meta.npz',ids=np.asarray([r['ID'] for r in rows]),Y=y,
            Y_train=smoothed,train_indices=np.flatnonzero(valid),slides=np.asarray([r['slide_id'] for r in rows]))
        (OUT/'img_native_512.partial.dat').rename(out)
        save_json(OUT/'manifest.json',dict(n=9000,matched_existing=matched,train_rows=int(valid.sum()),
            excluded_ids=[rows[i]['ID'] for i in np.flatnonzero(~valid)],
            exclusion='Same as validated OOF: exclude rows with any nonpositive metric before slide averaging.',
            source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            archive_sha256=rawmanifest['archive_sha256'],cache_bytes=out.stat().st_size,
            validation='No holdout here. Never report training-set accuracy as validation.'))
        print((OUT/'manifest.json').read_text(),flush=True)


if __name__=='__main__': main()
