#ifndef SPEXPLAIN_AVGPOOL_LAYER_H
#define SPEXPLAIN_AVGPOOL_LAYER_H

#include "NetworkLayer.h"

#include <array>
#include <cstddef>
#include <vector>

namespace spexplain {

// Average pooling. Unlike MaxPoolLayer this is an *affine* map: each output is the mean of its
// pooling window, so it introduces no branching and is encoded as a plain linear combination.
class AvgPoolLayer : public NetworkLayer {
public:
    using Kernel = std::array<std::size_t, 2>;   // {kH, kW}
    using Strides = std::array<std::size_t, 2>;  // {sH, sW}
    using Padding = std::array<std::size_t, 4>;  // {padTop,padBottom,padLeft,padRight}

    AvgPoolLayer(Shape inputShape, Kernel kernel, Strides strides, Padding padding, bool countIncludePad);

    Values computeLayerOutput(Values const & input) const override;

    Kernel const & getKernel() const { return kernel; }
    Strides const & getStrides() const { return strides; }
    Padding const & getPadding() const { return padding; }
    bool getCountIncludePad() const { return countIncludePad; }

    // For each (flattened) output element, the (flattened) input indices in its pooling window.
    // Padding cells contribute nothing and are simply absent from the window.
    std::vector<std::vector<std::size_t>> windowIndices() const;

    // Divisor of each output element: the window size when count_include_pad is 0 (padding cells
    // are excluded from the average), or the full kernel area when it is 1 (padding counts as 0).
    std::vector<std::size_t> windowDivisors() const;

private:
    static Shape makeOutputShape(Shape const & inputShape, Kernel const & kernel, Strides const & strides, Padding const & padding);

    Kernel kernel;
    Strides strides;
    Padding padding;
    bool countIncludePad;
};

} // namespace spexplain

#endif // SPEXPLAIN_AVGPOOL_LAYER_H
