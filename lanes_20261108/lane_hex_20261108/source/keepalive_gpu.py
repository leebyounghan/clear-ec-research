"""Last-resort keepalive for one GPU (HEX lane, 2026-11-11; generalised from keepalive_gpu1.py, GPU = CUDA_VISIBLE_DEVICES), only for gaps with nothing queued. About 35 % duty of fp16 matmuls on a
512 MB working set. It yields to any real job: every 30 s it reads that GPU's memory and compute-process list with nvidia-smi and
exits when another process holds more than 3 GB there or is listed on the card. usage: keepalive_gpu1.py [--minutes 0]"""
import os, sys, time, subprocess
import torch

IDX = int(os.environ["CUDA_VISIBLE_DEVICES"]); UUID = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(IDX)], capture_output=True, text=True, timeout=30).stdout.strip()   # physical GPU = CUDA_VISIBLE_DEVICES


def other_job(own_mb):
    q = lambda *a: subprocess.run(['nvidia-smi', f'--id={UUID}', *a, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30).stdout
    try:
        used = float(q('--query-gpu=memory.used').strip().splitlines()[0]); pids = [int(x) for x in q('--query-compute-apps=pid').split() if x.strip().isdigit()]
    except Exception:
        return False
    return used - own_mb > 3000 or any(p != os.getpid() for p in pids)


if __name__ == '__main__':
    minutes = float(sys.argv[sys.argv.index('--minutes') + 1]) if '--minutes' in sys.argv else 0.0
    assert UUID.startswith('GPU-'), UUID
    side = 9216; a = torch.randn(side, side, device='cuda', dtype=torch.float16); b = a.clone(); torch.cuda.synchronize()
    own = torch.cuda.memory_reserved() / 2 ** 20 + 600; t0 = time.time(); last = 0.0; duty = 0.35
    print(f'keepalive on {UUID}, own about {own:.0f} MB', flush=True)
    while True:
        if time.time() - last > 30:
            last = time.time()
            if other_job(own): print(time.strftime('%FT%TZ', time.gmtime()), 'another job on the GPU: exit', flush=True); break
            if minutes and time.time() - t0 > minutes * 60: print('time limit: exit', flush=True); break
        ts = time.time()
        for _ in range(4): b = a @ b; b = b / (b.abs().max() + 1)
        torch.cuda.synchronize(); w = time.time() - ts; time.sleep(w * (1 - duty) / duty)
