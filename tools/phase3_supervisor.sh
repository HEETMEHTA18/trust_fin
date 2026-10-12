#!/usr/bin/env bash
# Sequentially runs the given Phase-3 parts. GPU is never left idle between
# parts; a short cool-down runs if the GPU is hotter than 78 C.
# Safe to re-run after any interruption: finished experiments are skipped.
cd "/home/heet18/Downloads/Research Papers" || exit 1
LOG=supervisor.log
wait_idle() {
  while pgrep -f "[n]bconvert.*Phase3_TrustFin" >/dev/null 2>&1; do sleep 30; done
}
for part in "$@"; do
  wait_idle
  for i in 1 2 3 4 5 6 7 8; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    [ "$t" -lt 78 ] && break
    echo "[$(date +%a\ %H:%M)] cool-down: GPU ${t}C >= 78C, waiting 60s" >> "$LOG"
    sleep 60
  done
  echo "[$(date +%a\ %H:%M)] launching $part (GPU ${t}C)" >> "$LOG"
  pid=$(python3 tools/phase3_shard.py launch "$part" --mode full --run | sed -n 's/.*started pid=\([0-9]*\).*/\1/p')
  if [ -z "$pid" ]; then
    echo "[$(date +%a\ %H:%M)] $part LAUNCH FAILED — moving on" >> "$LOG"
    continue
  fi
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  echo "[$(date +%a\ %H:%M)] $part finished" >> "$LOG"
done
echo "[$(date +%a\ %H:%M)] ALL REQUESTED PARTS DONE" >> "$LOG"
