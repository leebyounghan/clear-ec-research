"""Last-resort GPU keepalive for the MLX notebook node.

The platform reclaims GPUs whose utilisation stays under ~10%, and a notebook
that is merely between experiments looks the same as an abandoned one. Prefer
`run_ablation_queue.sh` — real experiments we want anyway. Use this only for
the gaps: it burns a configurable duty cycle on every visible GPU and exits the
moment a real job (main_ppo / torchrun / sglang / vllm serve) appears, so it can never
compete with training for memory or SMs.

  python3 gpu_keepalive.py [--duty 0.35] [--minutes 0] [--mem-mb 512]

--minutes 0 runs until a real job shows up or the process is killed.
"""

import argparse
import subprocess
import time

import torch


def real_job_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-af", "main_ppo|torchrun|sglang.launch|eval_pixelgate|vllm serve|vllm.entrypoints"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    for line in out.splitlines():
        if "gpu_keepalive" in line or "pgrep" in line:
            continue
        if line.strip():
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ignore-jobs", action="store_true",
                    help="실제 잡이 돌아도 종료하지 않는다 — 캐스케이드의 CPU 구간처럼 GPU가 비는 틈을 메울 때")
    ap.add_argument("--duty", type=float, default=0.35,
                    help="fraction of wall time spent computing (0-1)")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="stop after this long; 0 = until a real job appears")
    ap.add_argument("--mem-mb", type=int, default=512,
                    help="working-set size per GPU")
    ap.add_argument("--poll", type=float, default=30.0,
                    help="seconds between real-job checks")
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("no CUDA devices visible")
    # side length so that three fp16 matrices fit the requested footprint
    side = int(((args.mem_mb * 1024 * 1024) / (3 * 2)) ** 0.5)
    side = max(1024, side - side % 256)
    mats = []
    for d in range(n):
        with torch.cuda.device(d):
            a = torch.randn(side, side, device=f"cuda:{d}", dtype=torch.float16)
            mats.append((a, a.clone()))
    print(f"[keepalive] {n} GPUs, {side}x{side} fp16, duty {args.duty:.0%}",
          flush=True)

    t0 = time.time()
    last_poll = 0.0
    # One duty cycle. Kept short so that a monitor sampling utilisation once a
    # second sees the average (~duty) rather than a 100%/0% square wave.
    period = 0.5
    while True:
        now = time.time()
        if args.minutes and (now - t0) / 60.0 >= args.minutes:
            print("[keepalive] time limit reached", flush=True)
            break
        if now - last_poll >= args.poll:
            last_poll = now
            if (not args.ignore_jobs and real_job_running()):
                print("[keepalive] real job detected — exiting", flush=True)
                break
        busy_until = time.time() + period * args.duty
        while time.time() < busy_until:
            for d, (a, b) in enumerate(mats):
                with torch.cuda.device(d):
                    torch.mm(a, b)
        for d in range(n):
            with torch.cuda.device(d):
                torch.cuda.synchronize()
        time.sleep(period * (1.0 - args.duty))


if __name__ == "__main__":
    main()
