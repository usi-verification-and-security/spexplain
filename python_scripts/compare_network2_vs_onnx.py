#!/usr/bin/env python3
"""Compare Network2 outputs against direct ONNX Runtime outputs on many inputs.

This tool runs an ONNX model in two ways for each generated input:
  1) Network2 executable (onnx-eval)
  2) ONNX Runtime direct inference

It reports max absolute differences and fails if any case exceeds tolerance.
"""

from __future__ import annotations

import argparse
import ast
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


def _resolve_shape_for_inference(shape: Sequence[object]) -> Tuple[List[int], bool]:
    """Resolve ONNX shape for one-sample inference.

    Returns:
      resolved_shape: concrete dimensions for ONNX Runtime input tensor
      has_batch: whether shape is treated as [batch, ...]
    """
    resolved: List[int] = []
    has_batch = False
    for idx, dim in enumerate(shape):
        if idx == 0 and (dim is None or isinstance(dim, str) or int(dim) == 1):
            resolved.append(1)
            has_batch = True
            continue
        if dim is None or isinstance(dim, str):
            raise ValueError(f"Unsupported dynamic dimension at axis {idx}: {dim!r}")
        resolved.append(int(dim))
    return resolved, has_batch


def _feature_shape(resolved_shape: Sequence[int], has_batch: bool) -> List[int]:
    if has_batch and len(resolved_shape) >= 1:
        return list(resolved_shape[1:])
    return list(resolved_shape)


def _tensor_size(shape: Sequence[int]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _vector_to_csv(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):.10g}" for v in values)


def _parse_output_list(text: str) -> List[float]:
    lines = [line for line in text.splitlines() if line.startswith("Output:")]
    if not lines:
        raise RuntimeError("Could not parse 'Output:' line from Network2 output")
    payload = lines[-1].split(":", 1)[1].strip()
    values = ast.literal_eval(payload)
    return [float(x) for x in values]


def _run_network2(network2_bin: Path, onnx_path: Path, flat_input: Sequence[float]) -> List[float]:
    csv = _vector_to_csv(flat_input)
    cmd = [str(network2_bin), str(onnx_path), csv]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Network2 execution failed (code={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _parse_output_list(proc.stdout)


def _run_onnxruntime(session, input_name: str, output_name: str, full_shape: Sequence[int], flat_input: Sequence[float]) -> List[float]:
    import numpy as np

    arr = np.asarray(flat_input, dtype=np.float32).reshape(full_shape)
    out = session.run([output_name], {input_name: arr})[0]
    return [float(x) for x in out.reshape(-1).tolist()]


def _generate_test_inputs(num_tests: int, feature_size: int, seed: int) -> List[List[float]]:
    import numpy as np

    if num_tests <= 0:
        raise ValueError("num_tests must be >= 1")

    tests: List[List[float]] = []

    # Deterministic coverage-oriented cases.
    tests.append([0.0] * feature_size)
    tests.append([1.0] * feature_size)
    tests.append([-1.0] * feature_size)
    tests.append(np.linspace(-1.0, 1.0, feature_size, dtype=np.float32).tolist())
    tests.append(np.linspace(1.0, -1.0, feature_size, dtype=np.float32).tolist())
    tests.append([1.0 if i % 2 == 0 else -1.0 for i in range(feature_size)])

    spike = [0.0] * feature_size
    spike[feature_size // 2] = 2.0
    tests.append(spike)

    # Random cases for broader coverage.
    rng = np.random.default_rng(seed)
    while len(tests) < num_tests:
        if len(tests) % 2 == 0:
            vec = rng.uniform(-2.0, 2.0, size=feature_size)
        else:
            vec = rng.normal(0.0, 1.0, size=feature_size)
        tests.append(vec.astype(np.float32).tolist())

    return tests[:num_tests]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path")
    parser.add_argument("network2_bin")
    parser.add_argument("--num-tests", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as exc:
        print(
            "Error: missing onnxruntime dependency. "
            "Install with: python3 -m pip install -r python_scripts/requirements-onnx-test.txt",
            file=sys.stderr,
        )
        print(f"Underlying import error: {exc}", file=sys.stderr)
        return 2

    onnx_path = Path(args.onnx_path)
    network2_bin = Path(args.network2_bin)
    if not onnx_path.exists():
        print(f"Error: ONNX file not found: {onnx_path}", file=sys.stderr)
        return 2
    if not network2_bin.exists():
        print(f"Error: Network2 executable not found: {network2_bin}", file=sys.stderr)
        return 2

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        print("Error: this test helper expects exactly 1 input and 1 output", file=sys.stderr)
        return 2

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    full_shape, has_batch = _resolve_shape_for_inference(input_info.shape)
    feat_shape = _feature_shape(full_shape, has_batch)
    feat_size = _tensor_size(feat_shape)

    tests = _generate_test_inputs(args.num_tests, feat_size, args.seed)

    print(f"Model: {onnx_path}")
    print(f"Input shape (full): {full_shape}")
    print(f"Input shape (Network2 semantic): {feat_shape}")
    print(f"Running {len(tests)} test inputs; tolerance={args.tol:g}; seed={args.seed}")

    worst_diff = -1.0
    worst_idx = -1
    failures = []

    for idx, test_input in enumerate(tests):
        n2 = _run_network2(network2_bin, onnx_path, test_input)
        ort_out = _run_onnxruntime(session, input_info.name, output_info.name, full_shape, test_input)

        if len(n2) != len(ort_out):
            failures.append((idx, math.inf, "length mismatch", n2, ort_out))
            continue

        max_abs_diff = 0.0
        for a, b in zip(n2, ort_out):
            max_abs_diff = max(max_abs_diff, abs(float(a) - float(b)))

        if max_abs_diff > worst_diff:
            worst_diff = max_abs_diff
            worst_idx = idx

        status = "PASS" if max_abs_diff <= args.tol else "FAIL"
        print(f"test[{idx:02d}] {status} max_abs_diff={max_abs_diff:.10g}")

        if max_abs_diff > args.tol:
            failures.append((idx, max_abs_diff, "tolerance exceeded", n2, ort_out))

    print("--- Summary ---")
    print(f"Total tests: {len(tests)}")
    print(f"Worst case: test[{worst_idx:02d}] diff={worst_diff:.10g}")
    print(f"Failures: {len(failures)}")

    if failures:
        idx, diff, reason, n2, ort_out = failures[0]
        print(f"First failure: test[{idx:02d}] reason={reason} diff={diff:.10g}", file=sys.stderr)
        print(f"Network2 output: {n2}", file=sys.stderr)
        print(f"ONNX output: {ort_out}", file=sys.stderr)
        return 5

    print("PASS: all generated inputs matched within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

