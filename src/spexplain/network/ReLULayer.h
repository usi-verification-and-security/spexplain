#ifndef SPEXPLAIN_RELU_LAYER_H
#define SPEXPLAIN_RELU_LAYER_H

#include "NetworkLayer.h"

namespace spexplain {

/// Point-wise ReLU activation layer.
/// Output shape equals input shape; each element is max(0, x).
class ReLULayer : public NetworkLayer {
public:
    /// \param shape  Both input and output have this shape.
    explicit ReLULayer(Shape shape)
        : NetworkLayer{"relu", shape, shape} {}

    Values computeLayerOutput(Values const & input) const override;
};

} // namespace spexplain

#endif // SPEXPLAIN_RELU_LAYER_H

