# Network2 FC ONNX Test Procedure

This procedure validates the `Network2` ONNX pipeline end-to-end for a fully connected model by running the same model in two ways and comparing outputs:

1. Parse ONNX model (`data/models/onnx/fc1.onnx`)
2. Build a `Network2` object
3. Evaluate an input vector with `Network2`
4. Evaluate the same input directly with ONNX Runtime
5. Compare both outputs with a tolerance check

## Python dependencies for direct ONNX run

```bash
python3 -m pip install -r python_scripts/requirements-onnx-test.txt
```

## Quick run

```bash
./data/scripts/test_scripts/test_network2_fc.sh
```

The script prints both outputs and a final `PASS`/`FAIL` comparison result.

Optional custom input (CSV format):

```bash
./data/scripts/test_scripts/test_network2_fc.sh 0.1,0.2,0.3,0.4
```

Optional tolerance:

```bash
TOL=1e-6 ./data/scripts/test_scripts/test_network2_fc.sh 0.1,0.2,0.3,0.4
```

Optional auto-install for missing Python deps:

```bash
AUTO_INSTALL_PY_DEPS=1 ./data/scripts/test_scripts/test_network2_fc.sh
```

## Direct binary run

After `make debug`, you can run:

```bash
./build-debug/onnx-eval ./data/models/onnx/fc1.onnx
```

Direct ONNX Runtime run:

```bash
python3 ./python_scripts/run_onnx_direct.py ./data/models/onnx/fc1.onnx
```


