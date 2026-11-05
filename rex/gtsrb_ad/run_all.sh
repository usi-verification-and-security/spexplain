#!/bin/bash
# Full pipeline: ReX+ConvAD explanations, then AD neuron activation files.
# Sample k in every output corresponds to row k of models/gtsrb_test_s100.csv.
set -u
PY=/Users/labbaf/miniconda3/envs/rex/bin/python
cd /Users/labbaf/Documents/rex/gtsrb_ad
N=${N:-100}
GAMMA=${GAMMA:-0.5}
for m in gtsrb_a gtsrb_b gtsrb_c gtsrb_d gtsrb_e; do
  echo "############ $m ############"
  $PY run_rex.py --model "$m" --n "$N" --gamma "$GAMMA" > "logs/rex_$m.log" 2>&1
  echo "rex exit=$? ; explanations:"
  grep -c "^  s.* area=" "logs/rex_$m.log" 2>/dev/null || true
  $PY neuron_report.py --model "$m" --masks "out/${m}_ad_g${GAMMA}_masks.pt"
  NA=/Users/labbaf/Documents/rex/neuron_activations
  $PY validate_format.py "$NA/${m}_ad_deactives.txt" \
      --complement "$NA/${m}_ad_actives.txt" "$NA/${m}_ad_deactives.txt"
  echo
done
echo "############ validating ############"
$PY validate_format.py /Users/labbaf/Documents/rex/neuron_activations/gtsrb_*_ad_deactives.txt \
                       /Users/labbaf/Documents/rex/neuron_activations/gtsrb_*_ad_actives.txt
echo "ALL DONE"
