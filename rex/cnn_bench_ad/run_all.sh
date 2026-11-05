#!/bin/bash
# ReX + ConvAD explanations for the NS and NAB cnn-bench families, then the AD neuron files
# and the comparison against the adversarial-search files.
# Sample k in every output is row k of data/datasets/gtsrb/gtsrb_s100_scaled.csv.
set -u
PY=${PY:-/Users/labbaf/miniconda3/envs/rex/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
MODELS=$REPO/data/models/gtsrb/cnn-bench
CSV=$REPO/data/datasets/gtsrb/gtsrb_s100_scaled.csv
N=${N:-100}
GAMMA=${GAMMA:-0.5}
cd "$HERE"
for m in NS1 NS2 NS3 NS4 NS5 NS6 NAB1 NAB2 NAB3 NAB4 NAB5 NAB6; do
  echo "############ $m ############"
  $PY run_rex.py --model "$MODELS/$m.onnx" --csv "$CSV" --n "$N" --gamma "$GAMMA" \
      > "logs/rex_$m.log" 2>&1
  echo "rex exit=$? ; explained: $(grep -c ' area=' "logs/rex_$m.log")"
  $PY neuron_files.py --model "$MODELS/$m.onnx" --masks "out/${m}_ad_g${GAMMA}_masks.pt"
done
$PY compare.py
echo "ALL DONE"
