# Network2 CNN ONNX Test Procedure

This procedure validates `Network2` on a CNN model (`data/models/onnx/conv.onnx`) by comparing:

1. `Network2` execution through `build-debug/onnx-eval`
2. Direct ONNX Runtime execution

It generates deterministic test inputs (coverage-oriented + random) and checks all outputs with a tolerance.

## Python dependencies

```bash
python3 -m pip install -r python_scripts/requirements-onnx-test.txt
```

## Run 50 tests (default)

```bash
./data/scripts/test_scripts/test_network2_cnn.sh
```

## Run with explicit test count / tolerance

```bash
./data/scripts/test_scripts/test_network2_cnn.sh 50 1e-5
```

## Optional seed override

```bash
SEED=11 ./data/scripts/test_scripts/test_network2_cnn.sh 50 1e-5
```

## Optional auto-install for missing Python deps

```bash
AUTO_INSTALL_PY_DEPS=1 ./data/scripts/test_scripts/test_network2_cnn.sh
```


