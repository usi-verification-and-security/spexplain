#ifndef SPEXPLAIN_RESHAPE_LAYER_H
#define SPEXPLAIN_RESHAPE_LAYER_H

#include "NetworkLayer.h"

namespace spexplain {

// ONNX Reshape (https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape) never permutes
// data, only regroups dimensions while preserving row-major order -- so, like Flatten, it is a
// pure re-indexing with no arithmetic: the flat value at position i means the same thing before
// and after, only the shape metadata used by later shape-aware layers (Conv, MaxPool, Transpose)
// changes. A genuine data permutation (e.g. NHWC -> NCHW for a multi-channel tensor) is a
// Transpose, not a Reshape; see TransposeLayer for that case.
class ReshapeLayer : public NetworkLayer {
public:
    ReshapeLayer(Shape inputShape, Shape outputShape)
        : NetworkLayer{"reshape", std::move(inputShape), std::move(outputShape)} {}

    Values computeLayerOutput(Values const & input) const override { return input; }
};

} // namespace spexplain

#endif // SPEXPLAIN_RESHAPE_LAYER_H
