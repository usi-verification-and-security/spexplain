#ifndef SPEXPLAIN_MAXPOOL_LAYER_H
#define SPEXPLAIN_MAXPOOL_LAYER_H

#include "NetworkLayer.h"

#include <array>
#include <cstddef>
#include <vector>

namespace spexplain {

class MaxPoolLayer : public NetworkLayer {
public:
    using Kernel = std::array<std::size_t, 2>;   // {kH, kW}
    using Strides = std::array<std::size_t, 2>;  // {sH, sW}
    using Padding = std::array<std::size_t, 4>;  // {padTop,padBottom,padLeft,padRight}

    MaxPoolLayer(Shape inputShape, Kernel kernel, Strides strides, Padding padding);

    Values computeLayerOutput(Values const & input) const override;

    Kernel const & getKernel() const { return kernel; }
    Strides const & getStrides() const { return strides; }
    Padding const & getPadding() const { return padding; }

    // For each (flattened) output element, the (flattened) input indices in its pooling window,
    // applying the same padding/bounds skipping as computeLayerOutput.
    std::vector<std::vector<std::size_t>> windowIndices() const;

private:
    static Shape makeOutputShape(Shape const & inputShape, Kernel const & kernel, Strides const & strides, Padding const & padding);

    Kernel kernel;
    Strides strides;
    Padding padding;
};

} // namespace spexplain

#endif // SPEXPLAIN_MAXPOOL_LAYER_H

