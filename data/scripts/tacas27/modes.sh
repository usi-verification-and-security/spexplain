## Settings of the runme.sh modes: which benchmarks, models, variants and experiments each mode
## runs, on which dataset and with which timeouts. Sourced by runme.sh, which sets MODE,
## DATA_DIR, ALL_VARIANTS and ALL_EXPERIMENTS.

MODES=(smoke quick short full)

NAMES=(heart_attack obesity loan mnist cifar gtsrb)

## Network widths, i.e. the <width>x prefix of the model files
WIDTHS=(50 200)

## Timeouts per explanation, in the order of NAMES
TIMEOUT_PERS=(      5m   5m   5m   10m  10m  14m)
SMOKE_TIMEOUT_PERS=(1m   1m   1m   5m   5m   5m)
QUICK_TIMEOUT_PERS=(200s 200s 200s 300s 400s 600s)

## <name> -> the dataset of the mode
## smoke, quick and short use the smaller datasets made by make_mode_datasets.py
## (quick and short contain the same samples), full uses the original one
function mode_dataset {
    local name=$1

    if [[ $MODE == full ]]; then
        printf "%s" "$DATA_DIR/datasets/$name/${name}_s100_scaled.csv"
    else
        printf "%s" "$DATA_DIR/datasets/$name/${name}_s100_scaled_$MODE.csv"
    fi
}

## <index in NAMES> -> the timeout per explanation of the mode
function mode_timeout_per {
    local i=$1

    case $MODE in
    smoke) printf "%s" ${SMOKE_TIMEOUT_PERS[$i]} ;;
    quick) printf "%s" ${QUICK_TIMEOUT_PERS[$i]} ;;
    *)     printf "%s" ${TIMEOUT_PERS[$i]} ;;
    esac
}

## Returns 1 if the mode skips the benchmark
function mode_includes_benchmark {
    local name=$1

    ## The smoke test skips the two slowest benchmarks
    [[ $MODE == smoke && $name =~ ^(cifar|gtsrb)$ ]] && return 1

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
        MODE_VARIANTS=(guide-all fix-all hybrid)
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

## Jobs left out of smoke, quick and short: the deep models where the variant mostly times out
## Returns 0 if the job is excluded
function is_excluded {
    local name=$1
    local width=$2
    local depth=$3
    local variant=$4

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
