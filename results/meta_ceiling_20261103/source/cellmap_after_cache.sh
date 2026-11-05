#!/bin/bash
# Waits for the cell-map cache, runs the CPU pre-test, then releases the queued cell-map job through its marker. A failed pre-test leaves no marker.
A=/workspace/clear-ec-research/experiments/meta_ceiling_20261103
until [ "$(tail -1 $A/sweep/cellmap_cache.log | cut -c1-4)" = "done" ]; do sleep 30; done
bash $A/precheck_cellmap.sh > $A/precheck_cellmap.log 2>&1
grep -q "passed cellmap" $A/precheck_cellmap.log && touch $A/sweep/cellmap_ready.marker
