SCRIPTS_DIR=$(realpath $(dirname $(dirname "$0")))
DATA_DIR=$(dirname "$SCRIPTS_DIR")
ROOT_DIR=$(dirname "$DATA_DIR")

# RUN_SCRIPT="$SCRIPTS_DIR/run-experiments.sh"
RUN_SCRIPT="$ROOT_DIR/sbatch.sh"

NAMES=(
heart_attack
obesity
mnist
cifar
gtsrb
)

KEYWORDS=(
HA50
OB50
MN50
CIF50
GTS50
)

TIMEOUT_PERS=(
5m
5m
10m
10m
15m
)

export CPU_PERCENTAGE=80

export EXPERIMENTS_SPEC=plain_itp
export VARIANTS_SPEC=neuron_activations

for i in ${!KEYWORDS[@]}; do
    KWD=${KEYWORDS[$i]}
    NAME="${NAMES[$i]}"
    TIMEOUT_PER=${TIMEOUT_PERS[$i]}

    declare -n lMODELS=${KWD}_MODELS
    lMODELS=("$DATA_DIR/models/$NAME/${NAME}_50x"*".nnet")

    DATASET="$DATA_DIR/datasets/$NAME/${NAME}_s100_scaled.csv"

    declare -n lARGS=${KWD}_ARGS
    lARGS=()
    for MODEL in "${lMODELS[@]}"; do
        lARGS+=("$MODEL")
        lARGS+=("$DATASET")
    done

    export TIMEOUT_PER
    "$RUN_SCRIPT" "${lARGS[@]}" $EXPERIMENTS_SPEC
done
