#!/usr/bin/env python3
"""Compare predictions of .h5, .onnx, and Network2 on a CSV dataset."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def load_dataset(csv_path: Path, label_col: Optional[str]) -> Tuple[List[str], List[List[float]]]:
    feature_names: List[str] = []
    features: List[List[float]] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError(f"Dataset has no header: {csv_path}")
        if label_col:
            if label_col not in fieldnames:
                raise RuntimeError(f"Dataset missing label column '{label_col}': {csv_path}")
            feature_names = [name for name in fieldnames if name != label_col]
        else:
            feature_names = list(fieldnames)

        for row in reader:
            features.append([float(row[name]) for name in feature_names])

    return feature_names, features


def parse_output_list(text: str) -> List[float]:
    lines = [line for line in text.splitlines() if line.startswith("Output:")]
    if not lines:
        raise RuntimeError("Could not parse 'Output:' line from Network2 output")
    values = ast.literal_eval(lines[-1].split(":", 1)[1].strip())
    return [float(v) for v in values]


def run_network2(onnx_eval: Path, onnx_path: Path, values: Sequence[float]) -> List[float]:
    csv_values = ",".join(f"{value:.10g}" for value in values)
    proc = subprocess.run(
        [str(onnx_eval), str(onnx_path), csv_values],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Network2 failed for {onnx_path.name} (code={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return parse_output_list(proc.stdout)


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) if a else 0.0


def compare_model(h5_path: Path, onnx_path: Path, onnx_eval: Path, xs: Sequence[Sequence[float]], tol: float) -> Tuple[float, float, float, int]:
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    model = tf.keras.models.load_model(str(h5_path), compile=False)

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    onnx_dtype = np.float64 if input_info.type == "tensor(double)" else np.float32

    worst_h5_onnx = -1.0
    worst_h5_n2 = -1.0
    worst_onnx_n2 = -1.0
    failures = 0

    for i, sample in enumerate(xs):
        h5_out = model(np.asarray([sample], dtype=np.float32), training=False).numpy().reshape(-1).tolist()
        onnx_out = session.run([output_info.name], {input_info.name: np.asarray([sample], dtype=onnx_dtype)})[0].reshape(-1).tolist()
        n2_out = run_network2(onnx_eval, onnx_path, sample)

        d_h5_onnx = max_abs_diff(h5_out, onnx_out)
        d_h5_n2 = max_abs_diff(h5_out, n2_out)
        d_onnx_n2 = max_abs_diff(onnx_out, n2_out)

        worst_h5_onnx = max(worst_h5_onnx, d_h5_onnx)
        worst_h5_n2 = max(worst_h5_n2, d_h5_n2)
        worst_onnx_n2 = max(worst_onnx_n2, d_onnx_n2)

        if max(d_h5_onnx, d_h5_n2, d_onnx_n2) > tol:
            failures += 1
            print(
                f"  row[{i:03d}] FAIL h5↔onnx={d_h5_onnx:.10g} h5↔n2={d_h5_n2:.10g} onnx↔n2={d_onnx_n2:.10g}",
                file=sys.stderr,
            )

    return worst_h5_onnx, worst_h5_n2, worst_onnx_n2, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--onnx-dir", default="", help="Optional directory with matching .onnx files (default: --models-dir)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--onnx-eval", required=True)
    parser.add_argument("--tol", type=float, default=5e-5)
    parser.add_argument("--pattern", default="*.h5", help="Glob pattern for selecting .h5 models")
    parser.add_argument("--label-col", default="output", help="Label column to exclude from features; use --all-columns to disable")
    parser.add_argument("--all-columns", action="store_true", help="Use every CSV column as model input features")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of dataset rows to compare")
    args = parser.parse_args()

    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tensorflow  # noqa: F401
    except Exception as exc:
        print(
            "Error: missing dependencies. Install with: python3 -m pip install -r python_scripts/requirements-heart-attack-compare.txt",
            file=sys.stderr,
        )
        print(f"Underlying import error: {exc}", file=sys.stderr)
        return 2

    models_dir = Path(args.models_dir)
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else models_dir
    dataset_path = Path(args.dataset)
    onnx_eval = Path(args.onnx_eval)

    if not models_dir.is_dir():
        print(f"Error: models directory not found: {models_dir}", file=sys.stderr)
        return 2
    if not dataset_path.is_file():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    if not onnx_dir.is_dir():
        print(f"Error: onnx directory not found: {onnx_dir}", file=sys.stderr)
        return 2
    if not onnx_eval.is_file():
        print(f"Error: onnx-eval not found: {onnx_eval}", file=sys.stderr)
        return 2

    label_col = None if args.all_columns else args.label_col
    feature_names, xs = load_dataset(dataset_path, label_col)
    if args.limit > 0:
        xs = xs[: args.limit]

    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(xs)}")
    print(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"Tolerance: {args.tol:g}")

    h5_files = sorted(models_dir.glob(args.pattern))
    if not h5_files:
        print(f"No models matched {args.pattern} in {models_dir}", file=sys.stderr)
        return 2

    total_failures = 0
    compared = 0
    for h5_path in h5_files:
        onnx_path = onnx_dir / f"{h5_path.stem}.onnx"
        if not onnx_path.exists():
            print(f"Skipping {h5_path.name}: missing {onnx_path.name}", file=sys.stderr)
            total_failures += 1
            continue

        compared += 1
        print(f"\nModel: {h5_path.name}")
        worst_h5_onnx, worst_h5_n2, worst_onnx_n2, failures = compare_model(h5_path, onnx_path, onnx_eval, xs, args.tol)
        print(f"  worst h5↔onnx : {worst_h5_onnx:.10g}")
        print(f"  worst h5↔n2   : {worst_h5_n2:.10g}")
        print(f"  worst onnx↔n2 : {worst_onnx_n2:.10g}")
        print(f"  row failures  : {failures}")
        total_failures += failures

    print("\n--- Summary ---")
    print(f"Compared models: {compared}")
    print(f"Total row failures: {total_failures}")

    if total_failures:
        print("FAIL: at least one model/sample exceeded tolerance", file=sys.stderr)
        return 5

    print("PASS: all .h5, .onnx, and Network2 outputs matched within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


