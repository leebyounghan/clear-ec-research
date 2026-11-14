#!/bin/bash
# alone_probe2.sh PID... : CPSAM s/update with my small jobs running, then with the given pids paused (SIGSTOP; each pid's
# command line must name source/gnn_hex.py or source/seg_run.py), then resumed (SIGCONT). Queue slots 0 for the whole probe,
# restored to 2 at the end. (HEX lane, 2026-11-08)
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; LOG=$L/logs/hexcps_f4.log; OUT=$L/logs/alone_probe.log
n() { grep -c '"status": "training"' $LOG; }
spd() { grep '"status": "training"' $LOG | tail -2 | python3 -c "
import sys, json
r=[json.loads(l) for l in sys.stdin]; print(round((r[1]['seconds_this_epoch']-r[0]['seconds_this_epoch'])/(r[1]['updates']-r[0]['updates']),3), 'epoch', r[1]['epoch'], 'updates', r[1]['updates'])"; }
echo 0 > $L/slots_gpu1.txt; echo "$(date -u +%T) slots 0 for the probe" >> $OUT
k=$(n); while [ $(n) -lt $((k + 2)) ] || [ $(n) -lt 3 ]; do sleep 20; done
echo "$(date -u +%T) with small jobs: $(spd) s/update" >> $OUT
P=""
for p in "$@"; do
  [ -r /proc/$p/cmdline ] || continue
  c=$(tr '\0' ' ' < /proc/$p/cmdline); case "$c" in *source/gnn_hex.py*|*source/seg_run.py*) P="$P $p";; esac
done
echo "$(date -u +%T) pausing:$P" >> $OUT; [ -n "$P" ] && kill -STOP $P
k=$(n); while [ $(n) -lt $((k + 2)) ]; do sleep 20; done
echo "$(date -u +%T) alone: $(spd) s/update" >> $OUT
[ -n "$P" ] && kill -CONT $P; echo 2 > $L/slots_gpu1.txt; echo "$(date -u +%T) resumed:$P, slots 2" >> $OUT
