#!/usr/bin/env python3
"""Run an ONNX model directly with onnxruntime and print the output.

Usage:
  python3 run_onnx_direct.py <onnx_path> [input_csv]

If input_csv is omitted, the script builds a deterministic default input of
0.1, 0.2, 0.3, ... matching the model's first input shape.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def parse_csv_input(csv: str) -> List[float]:
    return [float(token) for token in csv.split(",") if token.strip()]


def make_default_input(size: int) -> List[float]:
    return [float(i + 1) / 10.0 for i in range(size)]


def tensor_size(shape) -> int:
    size = 1
    for dim in shape:
        if dim in (None, "", "?") or isinstance(dim, str):
            raise ValueError(f"Dynamic or unknown dimension {dim!r} is not supported in this test helper")
        size *= int(dim)
    return size


def resolve_input_shapes(input_shape):
    """Resolve input shape for one-sample inference.

    Returns (full_shape, feature_shape), where feature_shape follows Network2
    semantics (batch axis omitted when present).
    """
    full_shape = []
    has_batch = False
    for idx, dim in enumerate(input_shape):
        if idx == 0 and (dim == 1 or dim is None or isinstance(dim, str)):
            full_shape.append(1)
            has_batch = True
            continue
        if dim in (None, "", "?") or isinstance(dim, str):
            raise ValueError(f"Dynamic or unknown dimension {dim!r} is not supported in this test helper")
        full_shape.append(int(dim))

    if has_batch:
        feature_shape = full_shape[1:]
    else:
        feature_shape = full_shape
    return full_shape, feature_shape


def format_values(values) -> str:
    return "[" + ", ".join(f"{float(v):.10g}" for v in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path")
    parser.add_argument("input_csv", nargs="?")
    args = parser.parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - error path is user-facing
        print(
            "Error: missing Python dependencies for direct ONNX execution. "
            "Install them with: python3 -m pip install -r python_scripts/requirements-onnx-test.txt",
            file=sys.stderr,
        )
        print(f"Underlying import error: {exc}", file=sys.stderr)
        return 2

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        print(f"Error: ONNX file not found: {onnx_path}", file=sys.stderr)
        return 2

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected exactly one model input, got {len(inputs)}")
    if len(outputs) != 1:
        raise RuntimeError(f"Expected exactly one model output, got {len(outputs)}")

    input_info = inputs[0]
    output_info = outputs[0]
    full_shape, feature_shape = resolve_input_shapes(list(input_info.shape))

    expected_size = tensor_size(feature_shape)
    if args.input_csv:
        values = parse_csv_input(args.input_csv)
    else:
        values = make_default_input(expected_size)

    if len(values) != expected_size:
        raise RuntimeError(f"Input size mismatch. Expected {expected_size} values, got {len(values)}")

    input_array = np.asarray(values, dtype=np.float32).reshape(full_shape)
    ort_inputs = {input_info.name: input_array}
    raw_output = session.run([output_info.name], ort_inputs)[0]
    output_values = raw_output.reshape(-1).tolist()

    if raw_output.ndim >= 1 and raw_output.shape[0] == 1:
        output_shape = list(raw_output.shape[1:])
    else:
        output_shape = list(raw_output.shape)

    print(f"ONNX direct runtime loaded: {onnx_path}")
    print(f"Input shape: {feature_shape}")
    print(f"Output shape: {output_shape}")
    print(f"Input: {format_values(values)}")
    print(f"Output: {format_values(output_values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

