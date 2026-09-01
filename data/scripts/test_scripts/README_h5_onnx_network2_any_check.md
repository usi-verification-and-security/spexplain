# Generic `.h5` / `.onnx` / `Network2` Output Check

This script compares predictions from:

1. Keras `.h5` model
2. ONNX Runtime on matching `.onnx` model
3. `Network2` evaluation through `onnx-eval`

for all selected samples in a CSV dataset.

## Script

- `data/scripts/test_scripts/check_h5_onnx_network2_any.sh`

## Requirements

```bash
python3 -m pip install -r python_scripts/requirements-heart-attack-compare.txt
```

## Basic usage

```bash
./data/scripts/test_scripts/check_h5_onnx_network2_any.sh \
  --models-dir ./data/models/heart_attack \
  --dataset ./data/datasets/heart_attack/heart_attack_full.csv \
  --label-col output \
  --tol 5e-5
```

## Check only one model

```bash
./data/scripts/test_scripts/check_h5_onnx_network2_any.sh \
  --models-dir ./data/models/heart_attack \
  --dataset ./data/datasets/heart_attack/heart_attack_full.csv \
  --label-col output \
  --pattern 'heart_attack-50.h5' \
  --limit 20 \
  --tol 5e-5
```

## Notes

- Matching ONNX files are assumed to have the same basename as each `.h5`.
- If your dataset has no label column, use `--all-columns`.
- If ONNX files are in a different folder, use `--onnx-dir <path>`.

