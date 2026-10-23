# Experiments with ONNX networks

The scripts in this folder run experiments with networks in the ONNX format (e.g. convolutional networks).
They run the frozen engine for ONNX networks, `build-onnx/engine` (see `make onnx`),
with exactly the same commands as in the previous experiments,
including the former option names of the engine (e.g. `--prefer-default-sample-neuron-activations`).
Hence, they use their own variants in `spec/variants` of this folder.

## `run1.sh` and `run-experiments.sh`

```
USAGE: ./run1.sh <onnx_model_fn> <dataset_fn> <exp_strategies_spec> [<name>] [<max_samples>] <args>...
USAGE: ./run-experiments.sh (<onnx_model_fn> <dataset_fn>)... <experiments_spec> [consecutive] [<max_samples>] [<filter_regex>] [-h|-n]
```

These are the ONNX counterparts of `../run1.sh` and `../run-experiments.sh`.
They take the same arguments and environment variables (including `VARIANT`/`VARIANTS_SPEC`, neuron-activation files, `OPTIONS`, `TIMEOUT` and `TIMEOUT_PER`) and produce the same output layout.
They run the `explain-onnx` action instead of `explain`.

Two things differ from the `.nnet` scripts:
* The model may live in a subdirectory of the dataset type. For example, `models/mnist/cnn-bench/S1.onnx` pairs with `datasets/mnist/mnist_s100_scaled.csv` and writes to `explanations/mnist/cnn-bench/S1/s100_scaled/<variant>`. Only the top-level type (`mnist`) has to match.
* ONNX files carry no input domain, so the bounds are resolved in this order:
  1. `INPUT_MIN`/`INPUT_MAX`, either one number for every feature or a comma-separated value per feature;
  2. the model's entry in `MODELS_DATASETS2_SPEC` (default `spec/models_datasets` of this folder);
  3. otherwise `[0,1]` for every feature, with a warning.

Further ONNX-only variables:
* `DROP_SIGMOID=true|false` (passed as `--drop-sigmoid`);
* `QUIET=0` to omit `--quiet`.

In directory `data/`:
```
M=(models/mnist/cnn-bench/{S1,NS1}.onnx)
INPUT_MIN=0 INPUT_MAX=1 VARIANTS_SPEC=neuron_activations ./scripts/cnn/run-experiments.sh ${M[@]/%/ datasets/mnist/mnist_s100_scaled.csv} plain_itp 10
```

## TACAS'27

The TACAS'27 CNN runs have wrapper scripts, `tacas27/{MN,GTS,CIF,IMN}.sh`, which work like the FC ones in `../tacas27/`.
See `tacas27/common` for the model and variant selection, and `tacas27/commands_list` for examples.
