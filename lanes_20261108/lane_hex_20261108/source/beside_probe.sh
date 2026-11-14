#!/bin/bash
# beside_probe.sh LIMIT : put the two mask-graph jobs at the top of jobs_gpu1.tsv, raise slots to 3 so the runner starts one
# beside CPSAM (two paused jobs hold the other slots), measure CPSAM s/update over 4 training lines; if above LIMIT, pause the
# new job (SIGSTOP, cmdline must name source/gnn_hex.py) and set slots back to 2. (HEX lane, 2026-11-08)
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; LOG=$L/logs/hexcps_f4.log; OUT=$L/logs/alone_probe.log; LIM=$1
n() { grep -c '"status": "training"' $LOG; }
spd() { grep '"status": "training"' $LOG | tail -4 | python3 -c "
import sys, json
r=[json.loads(l) for l in sys.stdin]; r=[q for q in r if q['epoch']==r[-1]['epoch']]
print(round((r[-1]['seconds_this_epoch']-r[0]['seconds_this_epoch'])/(r[-1]['updates']-r[0]['updates']),3))"; }
cd $L; { grep -P '^mask_' jobs_gpu1.tsv; grep -vP '^mask_' jobs_gpu1.tsv; } > jobs_gpu1.tsv.new; mv jobs_gpu1.tsv.new jobs_gpu1.tsv
echo 3 > slots_gpu1.txt; echo "$(date -u +%T) mask jobs to top, slots 3" >> $OUT
while [ ! -f logs/mask_hex_s5.log ]; do sleep 10; done; sleep 40
P=$(grep 'started mask_hex_s5 pid' logs/queue_gpu1.log | tail -1 | sed 's/.* pid \([0-9]*\) .*/\1/')
k=$(n); while [ $(n) -lt $((k + 4)) ]; do sleep 20; done
s=$(spd); echo "$(date -u +%T) beside mask_hex_s5 (pid $P): $s s/update (limit $LIM)" >> $OUT
if python3 -c "import sys; sys.exit(0 if $s > $LIM else 1)"; then
  c=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null); case "$c" in *source/gnn_hex.py*) kill -STOP $P; echo 2 > slots_gpu1.txt; echo "$(date -u +%T) over limit: paused $P, slots 2" >> $OUT;; esac
else echo "$(date -u +%T) within limit: keeping $P, slots 3" >> $OUT; fi
