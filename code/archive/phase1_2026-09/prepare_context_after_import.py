"""Bounded CPU continuation after the raw import publishes its validated manifest."""
import argparse
import json
from pathlib import Path
import time
from fullframe_views import prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='datasets/clear_ec_fullframe_20260904')
    parser.add_argument('--import-pid',type=int,required=True)
    args=parser.parse_args()
    root=Path(args.root); started=time.time()
    while not (root/'fullframe_manifest.json').exists():
        state=Path(f'/proc/{args.import_pid}/stat')
        if not state.exists() or state.read_text().split()[2]=='Z':
            raise RuntimeError('Raw importer exited without a validated manifest; inspect its log')
        if time.time()-started>2*3600:
            raise TimeoutError('raw import did not complete within two hours')
        time.sleep(10)
    prepare(root)
    print(json.dumps({'stage':'context_ready','root':str(root)},indent=2),flush=True)


if __name__=='__main__': main()
