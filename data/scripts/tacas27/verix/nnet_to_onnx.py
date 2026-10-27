"""
Usage
-----
  python nnet_to_onnx.py input.nnet [output.onnx]

"""

import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _skip_comments(lines):
    return [ln for ln in lines if not ln.strip().startswith("//")]


def parse_nnet(path: str) -> dict:
    with open(path, "r") as f:
        raw_lines = f.readlines()

    lines = _skip_comments(raw_lines)

    def parse_csv(line):
        return [x.strip() for x in line.split(",") if x.strip() != ""]

    dims = parse_csv(lines[0])
    num_layers  = int(dims[0])
    input_size  = int(dims[1])
    output_size = int(dims[2])

    layer_sizes = [int(x) for x in parse_csv(lines[1])]
    assert len(layer_sizes) == num_layers + 1, (
        f"Expected {num_layers + 1} layer sizes, got {len(layer_sizes)}"
    )
    assert layer_sizes[0] == input_size, "layer_sizes[0] must equal input_size"
    assert layer_sizes[-1] == output_size, "layer_sizes[-1] must equal output_size"

    symmetric = int(parse_csv(lines[2])[0])

    input_mins = np.array([float(x) for x in parse_csv(lines[3])], dtype=np.float32)

    input_maxs = np.array([float(x) for x in parse_csv(lines[4])], dtype=np.float32)

    means = np.array([float(x) for x in parse_csv(lines[5])], dtype=np.float32)

    ranges = np.array([float(x) for x in parse_csv(lines[6])], dtype=np.float32)

    assert len(means)  == input_size + 1, f"Expected {input_size+1} means, got {len(means)}"
    assert len(ranges) == input_size + 1, f"Expected {input_size+1} ranges, got {len(ranges)}"

    weights = []
    biases  = []
    line_idx = 7

    for layer_idx in range(num_layers):
        rows = layer_sizes[layer_idx + 1]
        cols = layer_sizes[layer_idx]

        W = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            row_vals = [float(x) for x in parse_csv(lines[line_idx])]
            assert len(row_vals) == cols, (
                f"Layer {layer_idx}, row {r}: expected {cols} weights, got {len(row_vals)}"
            )
            W[r] = row_vals
            line_idx += 1

        b = np.zeros(rows, dtype=np.float32)
        for r in range(rows):
            b[r] = float(parse_csv(lines[line_idx])[0])
            line_idx += 1

        weights.append(W)
        biases.append(b)

    return {
        "num_layers":  num_layers,
        "input_size":  input_size,
        "output_size": output_size,
        "layer_sizes": layer_sizes,
        "symmetric":   symmetric,
        "input_mins":  input_mins,
        "input_maxs":  input_maxs,
        "means":       means,
        "ranges":      ranges,
        "weights":     weights,
        "biases":      biases,
    }



def build_onnx_model(net: dict, opset_version: int = 13) -> onnx.ModelProto:
    """
    Build an ONNX model that includes:
      1. Input normalisation  : x_norm = (x - mean_in) / range_in
      2. Fully-connected + ReLU layers (ReLU omitted on final layer)
      3. Output de-normalisation: y = y_raw * range_out + mean_out
    """
    nodes       = []
    initializers = []

    input_size  = net["input_size"]
    output_size = net["output_size"]
    num_layers  = net["num_layers"]
    weights     = net["weights"]
    biases      = net["biases"]
    means       = net["means"]
    ranges      = net["ranges"]


    input_ranges = ranges[:input_size].copy()
    input_ranges[input_ranges == 0.0] = 1.0

    def add_init(name, arr):
        tensor = numpy_helper.from_array(arr, name=name)
        initializers.append(tensor)


    mean_in  = means[:input_size]
    range_in = input_ranges

    # The parser of Marabou supports neither Div nor Sub with broadcasting,
    # so the normalization is left out where it is the identity (added to the original script)
    if np.any(mean_in != 0) or np.any(range_in != 1):
        add_init("mean_in",  mean_in)
        add_init("range_in", range_in)

        nodes.append(helper.make_node(
            "Sub", inputs=["input", "mean_in"], outputs=["x_sub_mean"]
        ))
        nodes.append(helper.make_node(
            "Div", inputs=["x_sub_mean", "range_in"], outputs=["x_norm"]
        ))

        current = "x_norm"
    else:
        current = "input"


    for i in range(num_layers):
        W = weights[i]
        b = biases[i]

       
        Wt = W.T.copy()
        w_name = f"W_{i}"
        b_name = f"b_{i}"
        add_init(w_name, Wt)
        add_init(b_name, b)

        mm_out = f"mm_{i}"
        fc_out = f"fc_{i}"
        nodes.append(helper.make_node(
            "MatMul",
            inputs=[current, w_name],
            outputs=[mm_out],
        ))
        nodes.append(helper.make_node(
            "Add",
            inputs=[mm_out, b_name],
            outputs=[fc_out],
        ))

        if i < num_layers - 1:
            relu_out = f"relu_{i}"
            nodes.append(helper.make_node(
                "Relu", inputs=[fc_out], outputs=[relu_out]
            ))
            current = relu_out
        else:
            current = fc_out


    range_out = np.array([ranges[-1]], dtype=np.float32)
    mean_out  = np.array([means[-1]],  dtype=np.float32)

    if range_out[0] != 1 or mean_out[0] != 0:
        add_init("range_out", range_out)
        add_init("mean_out",  mean_out)

        nodes.append(helper.make_node(
            "Mul", inputs=[current, "range_out"], outputs=["y_scaled"]
        ))
        nodes.append(helper.make_node(
            "Add", inputs=["y_scaled", "mean_out"], outputs=["output"]
        ))
    else:
        nodes.append(helper.make_node(
            "Identity", inputs=[current], outputs=["output"]
        ))


    graph = helper.make_graph(
        nodes,
        "nnet_model",
        inputs=[helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [None, input_size]
        )],
        outputs=[helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [None, output_size]
        )],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset_version)],
    )
    model.ir_version = 8
    model.doc_string = (
        "Converted from NNet (SISL/Kyle Julian) format. "
        "Includes input normalisation and output de-normalisation."
    )

    onnx.checker.check_model(model)
    return model




def nnet_forward(net: dict, x: np.ndarray) -> np.ndarray:
    """
    Reference NumPy forward pass matching the NNet format spec.
    x shape: [batch, input_size]  (float32)
    """
    input_size = net["input_size"]
    means      = net["means"]
    ranges     = net["ranges"]

    input_ranges = ranges[:input_size].copy()
    input_ranges[input_ranges == 0.0] = 1.0

    h = (x - means[:input_size]) / input_ranges

    for i, (W, b) in enumerate(zip(net["weights"], net["biases"])):
        h = h @ W.T + b
        if i < net["num_layers"] - 1:
            h = np.maximum(h, 0.0)

    h = h * ranges[-1] + means[-1]
    return h.astype(np.float32)


def verify(net: dict, model: onnx.ModelProto, n_samples: int = 10, rtol=1e-4, atol=1e-5):
    """
    Run a quick numerical sanity check between the reference forward pass
    and the ONNX model.  Requires onnxruntime.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [verify] onnxruntime not installed; skipping numerical verification.")
        return

    sess = ort.InferenceSession(model.SerializeToString())
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    rng = np.random.default_rng(42)
    x = rng.uniform(
        net["input_mins"].astype(np.float64),
        net["input_maxs"].astype(np.float64),
        size=(n_samples, net["input_size"]),
    ).astype(np.float32)

    ref  = nnet_forward(net, x)
    onnx_out = sess.run([output_name], {input_name: x})[0]

    max_err = np.max(np.abs(ref - onnx_out))
    if np.allclose(ref, onnx_out, rtol=rtol, atol=atol):
        print(f"  [verify] PASSED  (max abs error = {max_err:.3e})")
    else:
        print(f"  [verify] FAILED  (max abs error = {max_err:.3e})")
        print("  First reference output:", ref[0])
        print("  First ONNX output:     ", onnx_out[0])




def convert(nnet_path: str, onnx_path=None, verify_result: bool = True):
    nnet_path = Path(nnet_path)
    if onnx_path is None:
        onnx_path = nnet_path.with_suffix(".onnx")
    else:
        onnx_path = Path(onnx_path)

    print(f"Reading  : {nnet_path}")
    net = parse_nnet(str(nnet_path))

    print(f"Layers   : {net['num_layers']}  |  "
          f"Layer sizes: {net['layer_sizes']}")
    print(f"Input    : {net['input_size']}  |  "
          f"Output: {net['output_size']}")

    print("Building ONNX graph …")
    model = build_onnx_model(net)

    print(f"Writing  : {onnx_path}")
    onnx.save(model, str(onnx_path))
    print(f"Done — {onnx_path.stat().st_size / 1024:.1f} KB")

    if verify_result:
        print("Verifying numerical correctness …")
        verify(net, model)

    return str(onnx_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    nnet_file = sys.argv[1]
    onnx_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert(nnet_file, onnx_file)
