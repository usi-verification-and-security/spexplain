# Network2 CNN MaxPool MNIST2 Test Procedure

This procedure validates `Network2` for `models/onnx/cnn_max_mnist2.onnx` by comparing:

1. `Network2` execution through `build-debug/onnx-eval`
2. Direct ONNX Runtime execution

If `cnn_max_mnist2.onnx` is not present, the script automatically falls back to
`cnn_max_mninst2.onnx`.

## Python dependencies

```bash
python3 -m pip install -r python_scripts/requirements-onnx-test.txt
```

## Run 50 tests (default)

```bash
./data/scripts/test_scripts/test_network2_cnn_max_mnist2.sh
```

## Run with explicit test count / tolerance

```bash
./data/scripts/test_scripts/test_network2_cnn_max_mnist2.sh 50 5e-5
```

## Optional seed override

```bash
SEED=11 ./data/scripts/test_scripts/test_network2_cnn_max_mnist2.sh 50 5e-5
```

## Optional auto-install for missing Python deps

```bash
AUTO_INSTALL_PY_DEPS=1 ./data/scripts/test_scripts/test_network2_cnn_max_mnist2.sh
```


