import os, signal, subprocess, time
L='/workspace/clear-ec-research/experiments/lane_gpu1_20261106'
# The short runner took its next line in the 4 seconds before the short queue was emptied. Stop that job too; it stays queued in the main queue.
for r in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
    if 'ssl_views_20261106.py cropxfer_both_f4' in r and 'bash -c' not in r:
        os.kill(int(r.split()[0]),signal.SIGTERM); print('TERM',r[:100])
time.sleep(6)
print(open(L+'/done_short.tsv').read())
print(repr(open(L+'/queue_short.tsv').read()))
print(subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1; nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader; ls '+L+'/ssl/cropxfer_both_f4/run '+L+'/ssl/cropopt_both_s1004_f4/run',shell=True,text=True))
