# Compare Heart Attack `.h5`, `.onnx`, and `Network2` Outputs

This check runs the same heart-attack dataset samples through three paths:

1. the original Keras `.h5` model,
2. the exported `.onnx` model through ONNX Runtime,
3. the `.onnx` model parsed and evaluated through `Network2`.

It reports the worst absolute differences for each model and fails if any sample
exceeds the chosen tolerance.

## Script

- `data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh`

## Requirements

```bash
python3 -m pip install -r python_scripts/requirements-heart-attack-compare.txt
```

## Full dataset check

```bash
./data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh 5e-5
```

## Quick smoke test on first 10 rows

```bash
./data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh 5e-5 10
```

## Auto-install missing dependencies

```bash
AUTO_INSTALL_PY_DEPS=1 ./data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh 5e-5
```

