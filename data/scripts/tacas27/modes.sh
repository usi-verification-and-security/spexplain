## Settings of the runme.sh modes: which benchmarks, models, variants and experiments each mode
## runs, on which dataset and with which timeouts. Sourced by runme.sh, which sets MODE,
## DATA_DIR, ALL_VARIANTS and ALL_EXPERIMENTS.

MODES=(smoke smokeLong quick short full)

## smokeLong: a smoke test for us, will be removed later. It is the smoke mode plus naive for the
## heart_attack 50x1-50x4 models and the unsat-core jobs, on the smoke dataset (and so into the
## same results folder s100_scaled_smoke/), with the smoke timeouts.

## The mode whose dataset, sample map and timeouts the mode uses
function dataset_mode {
    if [[ $MODE == smokeLong ]]; then
        printf "%s" smoke
    else
        printf "%s" $MODE
    fi
}

## Returns 0 for the smoke modes
function is_smoke_mode {
    [[ $MODE == smoke || $MODE == smokeLong ]]
}

NAMES=(heart_attack obesity loan mnist cifar gtsrb)

## The models <name>_<width>x<depth>.nnet: the widths, i.e. the neurons per hidden layer,
## and the depths, i.e. the hidden layers, the same for both widths
WIDTHS=(50 200)
TABULAR_DEPTHS=(1 2 3 4 6 8)
IMAGE_DEPTHS=(1 2 3 4 6 8 10 12 14)

## <name> -> the depths of its models
function model_depths {
    local name=$1

    case $name in
    heart_attack|obesity|loan) printf "%s" "${TABULAR_DEPTHS[*]}" ;;
    mnist|cifar|gtsrb)         printf "%s" "${IMAGE_DEPTHS[*]}" ;;
    esac
}

## Timeouts per explanation, in the order of NAMES
TIMEOUT_PERS=(      5m   5m   5m   10m  10m  14m)
SMOKE_TIMEOUT_PERS=(1m   1m   1m   5m   5m   5m)
QUICK_TIMEOUT_PERS=(200s 200s 200s 300s 400s 600s)

## <name> -> the dataset of the mode
## smoke, quick and short use the smaller datasets made by make_mode_datasets.py
## (short contains the samples of quick first, then more), full uses the original one
function mode_dataset {
    local name=$1

    if [[ $MODE == full ]]; then
        printf "%s" "$DATA_DIR/datasets/$name/${name}_s100_scaled.csv"
    else
        printf "%s" "$DATA_DIR/datasets/$name/${name}_s100_scaled_$(dataset_mode).csv"
    fi
}

## <index in NAMES> -> the timeout per explanation of the mode
function mode_timeout_per {
    local i=$1

    case $(dataset_mode) in
    smoke) printf "%s" ${SMOKE_TIMEOUT_PERS[$i]} ;;
    quick) printf "%s" ${QUICK_TIMEOUT_PERS[$i]} ;;
    *)     printf "%s" ${TIMEOUT_PERS[$i]} ;;
    esac
}

## Returns 1 if the mode skips the benchmark
function mode_includes_benchmark {
    local name=$1

    ## The smoke test skips the two slowest benchmarks
    is_smoke_mode && [[ $name =~ ^(cifar|gtsrb)$ ]] && return 1

    return 0
}

## Sets MODE_VARIANTS and MODE_EXPERIMENTS of the mode for the width
## Returns 1 if the mode skips the width
function set_mode_for_width {
    local width=$1

    MODE_VARIANTS=("${ALL_VARIANTS[@]}")
    MODE_EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")

    case $MODE in
    smoke)
        (( $width == 50 )) || return 1
        MODE_VARIANTS=(guide-all fix-all hybrid fix-all-no-vars)
        MODE_EXPERIMENTS=(itp_aweak_bstrong)
        ;;
    smokeLong)
        (( $width == 50 )) || return 1
        MODE_VARIANTS=(naive guide-all fix-all hybrid fix-all-no-vars)
        MODE_EXPERIMENTS=(itp_aweak_bstrong)
        ;;
    short|quick)
        (( $width == 50 )) || MODE_EXPERIMENTS=(itp_aweak_bstrong)
        ;;
    esac

    _check_includes ALL_VARIANTS MODE_VARIANTS
    _check_includes ALL_EXPERIMENTS MODE_EXPERIMENTS

    return 0
}

## Jobs left out of the modes; returns 0 if the job is excluded
function is_excluded {
    local name=$1
    local width=$2
    local depth=$3
    local variant=$4

    ## In all modes: fix-all-no-vars only for the 50x1-50x4 models of heart_attack, loan and mnist
    [[ $variant == fix-all-no-vars ]] && {
        [[ $name =~ ^(heart_attack|loan|mnist)$ ]] && (( $width == 50 && $depth <= 4 )) && return 1
        return 0
    }

    ## In smoke and smokeLong: only the models up to 50x6
    is_smoke_mode && (( $depth > 6 )) && return 0

    ## In smokeLong: naive only for the 50x1-50x4 models of heart_attack (smoke runs no naive)
    [[ $MODE == smokeLong && $variant == naive ]] && {
        [[ $name == heart_attack ]] && (( $depth <= 4 )) && return 1
        return 0
    }

    ## In smoke, quick and short: the deep models where the variant mostly times out
    [[ $MODE == full ]] && return 1

    case $variant in
    naive)
        case $name in
        heart_attack|obesity|loan)
            (( $width == 50 && $depth >= 8 )) && return 0
            ;;
        mnist)
            (( $width == 50 && $depth >= 6 || $width == 200 && $depth >= 3 )) && return 0
            ;;
        cifar|gtsrb)
            (( $width == 50 && $depth >= 4 || $width == 200 && $depth >= 3 )) && return 0
            ;;
        esac
        ;;
    guide-all)
        case $name in
        cifar|gtsrb)
            (( $width == 200 && $depth >= 4 )) && return 0
            ;;
        esac
        ;;
    esac

    return 1
}

## collect-stats: only these jobs (regex over the job ids <name>/<width>x<depth>/<variant>/<experiment>):
## naive, guide-all, fix-all and hybrid, the itp experiments only (no fix-all-no-vars, no unsat cores)
STATS_FILTER=${STATS_FILTER:-'^[^/]+/[^/]+/(naive|guide-all|fix-all|hybrid)/itp_'}

## The CNN (ONNX) models data/models/<name>/<CNN_MODEL_SET>/<family><depth>.onnx, run with the frozen
## engine for ONNX networks (make onnx) by ../cnn/run1.sh, on the same datasets and with the same
## timeouts as the .nnet models of the benchmark (mode_dataset, mode_timeout_per)
CNN_NAMES=(mnist cifar gtsrb)
CNN_MODEL_SET=cnn-bench
CNN_FAMILIES=(NS NAB)
CNN_DEPTHS=(1 2 3 4 5 6)
## The variants are of ../cnn/spec/variants/all, the experiments of EXPERIMENTS_SPEC
CNN_VARIANTS=(naive guide-all fix-all hybrid)
CNN_EXPERIMENTS=(itp_aweak_bstrong)
## All CNN-bench models take pixels in [0,1]; exported for the jobs of GNU parallel
export CNN_INPUT_MIN=0
export CNN_INPUT_MAX=1

## Sets MODE_CNN_VARIANTS and MODE_CNN_EXPERIMENTS of the mode
function set_cnn_mode {
    MODE_CNN_VARIANTS=("${CNN_VARIANTS[@]}")
    MODE_CNN_EXPERIMENTS=("${CNN_EXPERIMENTS[@]}")

    is_smoke_mode && MODE_CNN_VARIANTS=(guide-all fix-all hybrid)

    _check_includes CNN_VARIANTS MODE_CNN_VARIANTS
    _check_includes ALL_EXPERIMENTS MODE_CNN_EXPERIMENTS
}

## CNN jobs left out of the modes; returns 0 if the job is excluded
function is_cnn_excluded {
    local name=$1
    local family=$2
    local depth=$3
    local variant=$4

    ## In smoke and smokeLong: only the models up to NS4 and NAB4
    is_smoke_mode && (( $depth > 4 )) && return 0

    return 1
}

## Unsat cores of interpolants, in all modes but smoke, besides the experiments of the mode:
## 'itp aweak, bstrong; ucore min' (the interpolant is computed in the same run, then reduced)
## of heart_attack 50x1, naive and fix-all. They run with OpenSMT (-V opensmt).
UCORE_EXPERIMENTS=(ucore_min_itp_aweak_bstrong)
## The experiments collection that contains them, for collect_stats.sh
export UCORE_EXPERIMENTS_SPEC=ucore_itp

## <name> <width> <depth> <variant> -> the ucore experiments of the job's model and variant
function ucore_experiments {
    local name=$1
    local width=$2
    local depth=$3
    local variant=$4

    [[ $MODE == smoke ]] && return 0

    [[ $name == heart_attack && $variant =~ ^(naive|fix-all)$ ]] && (( $width == 50 && $depth == 1 )) && {
        printf "%s" "${UCORE_EXPERIMENTS[*]}"
    }
    return 0
}

## Returns 0 if the experiment is one of UCORE_EXPERIMENTS
function is_ucore_experiment {
    [[ " ${UCORE_EXPERIMENTS[*]} " == *" $1 "* ]]
}

## The timeout per explanation of the ucore experiments, in all modes (as in the ucore table)
function mode_ucore_timeout_per {
    printf "%s" 20m
}

## <name> <original sample> -> its index in the dataset of the mode (both 1-based),
## from data/datasets/mode_sample_maps.csv; returns 1 if the dataset lacks the sample
function mode_sample {
    local name=$1
    local orig=$2

    awk -F, -v name=$name -v mode=$(dataset_mode) -v orig=$orig '
        $1 == name && $2 == mode && $4 == orig { print $3; found = 1; exit }
        END { exit !found }
    ' "$DATA_DIR/datasets/mode_sample_maps.csv"
}
