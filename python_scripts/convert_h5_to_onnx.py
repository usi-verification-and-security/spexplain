#!/usr/bin/env python3
"""Convert Keras .h5 models in a directory to ONNX files.

The ONNX files are written next to each source model, with the same basename.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert_model(h5_path: Path, overwrite: bool) -> Path:
    try:
        import numpy as np
        import tensorflow as tf
        import tf2onnx
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies for H5->ONNX conversion. Install with: "
            "python3 -m pip install tensorflow tf2onnx"
        ) from exc

    onnx_path = h5_path.with_suffix(".onnx")
    if onnx_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {onnx_path}")

    model = tf.keras.models.load_model(str(h5_path), compile=False)

    # Warm-up call so Keras can export models that are loaded but not yet called.
    sample_inputs = []
    for inp in model.inputs:
        shape = [dim if dim is not None else 1 for dim in inp.shape]
        sample_inputs.append(np.zeros(shape, dtype=np.float32))
    _ = model(sample_inputs[0] if len(sample_inputs) == 1 else sample_inputs)

    # Preferred path for Keras 3 + TF backend.
    try:
        model.export(str(onnx_path), format="onnx")
        return onnx_path
    except Exception:
        pass

    # Fallback path for environments where model.export(onnx) is unavailable.
    specs = []
    for inp in model.inputs:
        shape = [dim if dim is not None else None for dim in inp.shape]
        specs.append(tf.TensorSpec(shape=shape, dtype=inp.dtype, name=inp.name.split(":")[0]))

    _model_proto, _external = tf2onnx.convert.from_keras(
        model,
        input_signature=tuple(specs),
        opset=13,
        output_path=str(onnx_path),
    )

    return onnx_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing .h5 models")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .onnx files")
    parser.add_argument("--pattern", default="*.h5", help="Glob pattern to select models (default: *.h5)")
    args = parser.parse_args()

    model_dir = Path(args.dir)
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"Error: not a directory: {model_dir}", file=sys.stderr)
        return 2

    h5_files = sorted(model_dir.glob(args.pattern))
    if not h5_files:
        print(f"No models matched pattern '{args.pattern}' in {model_dir}")
        return 0

    print(f"Found {len(h5_files)} model(s) in {model_dir}")

    failures = []
    for h5_path in h5_files:
        try:
            onnx_path = convert_model(h5_path, overwrite=args.overwrite)
            print(f"OK   {h5_path.name} -> {onnx_path.name}")
        except Exception as exc:
            failures.append((h5_path, str(exc)))
            print(f"FAIL {h5_path.name}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} conversion(s) failed.", file=sys.stderr)
        return 1

    print("\nAll conversions completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


