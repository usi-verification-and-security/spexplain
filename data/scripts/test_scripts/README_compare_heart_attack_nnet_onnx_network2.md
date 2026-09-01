# Compare `heart_attack-50.nnet` vs `heart_attack-50.onnx` (`Network2`)

This comparison runs every sample in:

- `data/datasets/heart_attack/heart_attack_full.csv`

through:

1. `Network::fromNNetFile(...)->evaluate(...)`
2. `Network2::fromONNXFile(...)->evaluate(...)`

and compares:

- output values
- classifications
- per-sample hidden-layer input/output values (`same` / `different`)

Example section of output:

```text
sample 1:
Layer 1:
input_values: same|different
output_values: same|different
```

## Script

- `data/scripts/test_scripts/compare_heart_attack_nnet_vs_onnx_network2.sh`

## Run full dataset

```bash
./data/scripts/test_scripts/compare_heart_attack_nnet_vs_onnx_network2.sh 5e-5 0
```

## Run subset (smoke test)

```bash
./data/scripts/test_scripts/compare_heart_attack_nnet_vs_onnx_network2.sh 5e-5 30
```

## Notes on summary fields

- `value mismatches raw`: direct `nnet` output vs `onnx/network2` output
- `value mismatches preSig`: `nnet` output vs **pre-sigmoid** ONNX value from `Network2` hidden activation inputs
- `value mismatches aligned`: after applying final sigmoid alignment to nnet when ONNX model ends with sigmoid
- `class mismatches native`: labels as returned directly by each implementation
- `class mismatches aligned`: labels derived from aligned output values (binary uses threshold 0.5)

