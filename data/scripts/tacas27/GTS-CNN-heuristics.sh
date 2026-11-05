#!/bin/bash
## Hybrid mode on GTSRB CNN-bench models with different neuron-fixing heuristics and baselines:
##   hybrid-adversarial   data/neuron_activations/gtsrb/cnn-bench/<model>.txt (PGD phase-stable neurons)
##   hybrid-rex-active    rex/cnn_bench_ad/neuron_activations/<model>_ad_actives.txt
##   hybrid-rex-deactive  rex/cnn_bench_ad/neuron_activations/<model>_ad_deactives.txt
##   hybrid-random        data/activations/gtsrb/cnn-bench/random/<model>.txt
##                        (random, of the same size per sample and layer as the adversarial set)
##   hybrid-negative      data/activations/gtsrb/cnn-bench/negative/<model>.txt (pre-activation < 0)
##   hybrid-positive      data/activations/gtsrb/cnn-bench/positive/<model>.txt (pre-activation > 0)
## The random/negative/positive files are made by rex/cnn_bench_ad/make_baseline_files.py.
## Only the itp_aweak_bstrong experiment is run. Each variant writes to its own directory
## data/explanations/gtsrb/cnn-bench/<model>/s100_scaled/<variant>/.
##
## Same usage and environment as ./common-cnn; the defaults here are
##   MODEL_IDS=NS1-4 and NAB1-4, VARIANTS_SPEC=hybrid_heuristics, EXPERIMENTS_SPEC=aweak_bstrong,
##   MAX_SAMPLES=20, TIMEOUT_PER=15m.
## Local pilot:  MAX_SAMPLES=1 TIMEOUT_PER=2m ./GTS-CNN-heuristics.sh
## Cluster:      WRAPPER=~/srun_commands/sbatch.sh ./GTS-CNN-heuristics.sh

NAME=gtsrb
TIMEOUT_PER=${TIMEOUT_PER:-15m}
MAX_SAMPLES=${MAX_SAMPLES:-20}
MODEL_IDS=${MODEL_IDS:-"NS1 NAB1 NS2 NAB2 NS3 NAB3 NS4 NAB4"}
VARIANTS_SPEC=${VARIANTS_SPEC:-hybrid_heuristics}
EXPERIMENTS_SPEC=${EXPERIMENTS_SPEC:-aweak_bstrong}

source "$(dirname "$(realpath "$0")")/common-cnn"
