# Convert Heart Attack `.h5` Models to ONNX

This utility converts all `.h5` models in `data/models/heart_attack` to `.onnx`
with the same basename and saves them in the same directory.

## Script

- `data/scripts/test_scripts/convert_heart_attack_h5_to_onnx.sh`

## Python requirements

```bash
python3 -m pip install -r python_scripts/requirements-h5-to-onnx.txt
```

## Convert all models

```bash
./data/scripts/test_scripts/convert_heart_attack_h5_to_onnx.sh
```

## Overwrite existing `.onnx`

```bash
./data/scripts/test_scripts/convert_heart_attack_h5_to_onnx.sh --overwrite
```

## Auto-install missing dependencies

```bash
AUTO_INSTALL_PY_DEPS=1 ./data/scripts/test_scripts/convert_heart_attack_h5_to_onnx.sh --overwrite
```

