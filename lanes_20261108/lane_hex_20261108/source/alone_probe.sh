#!/bin/bash
# alone_probe.sh : CPSAM s/update with my small jobs running, then with them paused (SIGSTOP, own pids, command line checked),
# then resume (SIGCONT). Queue slots set to 0 during the pause and restored after. (HEX lane, 2026-11-08)
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; LOG=$L/logs/hexcps_f4.log; OUT=$L/logs/alone_probe.log
n() { grep -c '"status": "training"' $LOG; }
spd() { grep '"status": "training"' $LOG | tail -2 | python3 -c "
import sys, json
r=[json.loads(l) for l in sys.stdin]; print(round((r[1]['seconds_this_epoch']-r[0]['seconds_this_epoch'])/(r[1]['updates']-r[0]['updates']),3), 'epoch', r[1]['epoch'], 'updates', r[1]['updates'])"; }
while [ $(n) -lt 3 ]; do sleep 20; done
echo "$(date -u +%T) with small jobs: $(spd) s/update" >> $OUT
echo 0 > $L/slots_gpu1.txt; P=""
for p in $(pgrep -f 'source/gnn_hex.py|source/seg_run.py'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline); case "$c" in *source/gnn_hex.py*|*source/seg_run.py*) P="$P $p";; esac
done
echo "$(date -u +%T) pausing:$P" >> $OUT; kill -STOP $P
k=$(n); while [ $(n) -lt $((k + 2)) ]; do sleep 20; done
echo "$(date -u +%T) alone: $(spd) s/update" >> $OUT
kill -CONT $P; echo 2 > $L/slots_gpu1.txt; echo "$(date -u +%T) resumed:$P, slots 2" >> $OUT
