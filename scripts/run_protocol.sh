#!/bin/bash
# Run the full Role 4 evaluation protocol for one dataset.
#
# Usage (from the repo root):
#   bash scripts/run_protocol.sh NAME TRAIN_FILE DEV_FILE TEST_FILE dev    # develop and inspect
#   bash scripts/run_protocol.sh NAME TRAIN_FILE DEV_FILE TEST_FILE test   # final, once
#
# Each phase runs three sweeps in parallel:
#   lambda sweep (5 FQI seeds), cost sweep with abstention (lambda=0.25),
#   cost sweep without abstention (lambda=0).
# The test phase refuses to run before the dev phase, and refuses to run twice.
set -euo pipefail

if [ $# -lt 5 ]; then
  echo "usage: $0 NAME TRAIN_FILE DEV_FILE TEST_FILE dev|test"; exit 1
fi
NAME=$1; TR=$2; DV=$3; TE=$4; PHASE=$5
OUT="outputs/rl/$NAME"
mkdir -p "$OUT"

PY="python3"
command -v caffeinate >/dev/null 2>&1 && PY="caffeinate -i python3"   # keep a Mac awake

case "$PHASE" in
  dev)
    EVAL="$DV"; TAG="dev"
    echo "== integrity check =="
    for f in "$TR" "$DV" "$TE"; do python3 rl/tree_data.py "$f"; done
    ;;
  test)
    EVAL="$TE"; TAG="TEST"
    if [ ! -f "$OUT/.dev_done" ]; then
      echo "Run the dev phase first."; exit 1
    fi
    if [ -f "$OUT/.test_done" ]; then
      echo "The test phase has already been run for $NAME. Not running it again."; exit 1
    fi
    ;;
  *)
    echo "PHASE must be dev or test"; exit 1
    ;;
esac

echo "== $NAME: $TAG phase, three sweeps in parallel (logs in $OUT) =="
$PY -m rl.sweep --train "$TR" --dev "$EVAL" --seeds 0,1,2,3,4 \
  --out "$OUT/${TAG}_lambda" > "$OUT/${TAG}_lambda.log" 2>&1 &
$PY -m rl.sweep --train "$TR" --dev "$EVAL" --lams 0.25 --cost_scales 0,0.5,1,2,4 --seeds 0,1,2 \
  --out "$OUT/${TAG}_cost" > "$OUT/${TAG}_cost.log" 2>&1 &
$PY -m rl.sweep --train "$TR" --dev "$EVAL" --no_abstain --lams 0 --cost_scales 0,0.5,1,2,4 \
  --seeds 0,1,2 --out "$OUT/${TAG}_cost_noabst" > "$OUT/${TAG}_cost_noabst.log" 2>&1 &
wait

echo "== summary: FQI vs threshold per setting =="
for log in "$OUT/${TAG}_lambda.log" "$OUT/${TAG}_cost.log" "$OUT/${TAG}_cost_noabst.log"; do
  echo "--- $(basename "$log")"
  grep -E "^lambda=|^  FQI  |^  threshold " "$log" || true
done

if [ "$PHASE" = dev ]; then
  touch "$OUT/.dev_done"
else
  touch "$OUT/.test_done"
  mkdir -p "results/$NAME"
  cp "$OUT"/dev_*.csv "$OUT"/dev_*.png "$OUT"/TEST_*.csv "$OUT"/TEST_*.png "results/$NAME"/
  echo "Copied CSVs and plots to results/$NAME"
fi
echo "done."
