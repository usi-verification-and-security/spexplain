# Network2 Test Scripts

This folder contains scripts that compare `Network2` inference against direct ONNX Runtime inference.

## Scripts

- `test_network2_any.sh` - unified runner for any supported ONNX model (single input/output)
- `test_network2_fc.sh` - focused FC test on `data/models/onnx/fc1.onnx`
- `test_network2_cnn.sh` - focused CNN test on `data/models/onnx/conv.onnx`
- `test_network2_cnn_max_mnist2.sh` - focused test for `cnn_max_mnist2`/`cnn_max_mninst2`
- `convert_heart_attack_h5_to_onnx.sh` - converts all `data/models/heart_attack/*.h5` to `.onnx` in the same folder
- `check_heart_attack_h5_onnx_network2.sh` - compares outputs of heart_attack `.h5`, `.onnx`, and `Network2` on the full CSV dataset
- `check_h5_onnx_network2_any.sh` - generic checker for any CSV dataset + matching `.h5`/`.onnx` model directory
- `compare_heart_attack_nnet_vs_onnx_network2.sh` - compares `Network` (.nnet) and `Network2` (.onnx) outputs/classifications on heart_attack dataset
- `compare_psi_onnx_vs_nnet.sh` - compares the SMT-LIB2 encodings produced by `encode-onnx` (Network2 + OpenSMTVerifier2) and `dump-psi` (Network + OpenSMTVerifier)
- `psi_diff.py` - structural comparison of two SMT-LIB2 psi files; matches assertion conjuncts by shape and compares numeric literals with a tolerance (used by `compare_psi_onnx_vs_nnet.sh`, also usable standalone)

## Model-specific docs

- `README_network2_fc_test.md`
- `README_network2_cnn_test.md`
- `README_network2_cnn_max_mnist2_test.md`
- `README_h5_to_onnx_heart_attack.md`
- `README_heart_attack_h5_onnx_network2_check.md`
- `README_h5_onnx_network2_any_check.md`
- `README_compare_heart_attack_nnet_onnx_network2.md`
- `README_compare_psi_onnx_vs_nnet.md`
