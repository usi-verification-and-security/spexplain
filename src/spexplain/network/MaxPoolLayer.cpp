#include "MaxPoolLayer.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace spexplain {

MaxPoolLayer::Shape MaxPoolLayer::makeOutputShape(Shape const & inputShape,
                                                   Kernel const & kernel,
                                                   Strides const & strides,
                                                   Padding const & padding)
{
    // inputShape = {C, H, W}
    std::size_t outH = (inputShape[1] + padding[0] + padding[1] - kernel[0]) / strides[0] + 1;
    std::size_t outW = (inputShape[2] + padding[2] + padding[3] - kernel[1]) / strides[1] + 1;
    return {inputShape[0], outH, outW};
}

MaxPoolLayer::MaxPoolLayer(Shape inputShape, Kernel kernel, Strides strides, Padding padding)
    : NetworkLayer{"maxpool", inputShape, makeOutputShape(inputShape, kernel, strides, padding)},
      kernel{kernel}, strides{strides}, padding{padding}
{
}

std::vector<std::vector<std::size_t>> MaxPoolLayer::windowIndices() const
{
    assert(inputShape.size() == 3);

    std::size_t const channels = inputShape[0];
    std::size_t const inH = inputShape[1];
    std::size_t const inW = inputShape[2];

    std::size_t const outH = outputShape[1];
    std::size_t const outW = outputShape[2];

    std::vector<std::vector<std::size_t>> windows(getOutputSize());

    for (std::size_t c = 0; c < channels; ++c)
    {
        for (std::size_t oh = 0; oh < outH; ++oh)
        {
            for (std::size_t ow = 0; ow < outW; ++ow)
            {
                std::size_t const outIdx = c * outH * outW + oh * outW + ow;
                auto & window = windows[outIdx];

                std::size_t startH = oh * strides[0];
                std::size_t startW = ow * strides[1];

                for (std::size_t kh = 0; kh < kernel[0]; ++kh)
                {
                    for (std::size_t kw = 0; kw < kernel[1]; ++kw)
                    {
                        std::size_t ihPadded = startH + kh;
                        std::size_t iwPadded = startW + kw;

                        if (ihPadded < padding[0] || iwPadded < padding[2])
                            continue;

                        std::size_t ih = ihPadded - padding[0];
                        std::size_t iw = iwPadded - padding[2];
                        if (ih >= inH || iw >= inW)
                            continue;

                        window.push_back(c * inH * inW + ih * inW + iw);
                    }
                }
            }
        }
    }

    return windows;
}

NetworkLayer::Values MaxPoolLayer::computeLayerOutput(Values const & input) const
{
    assert(input.size() == getInputSize());
    assert(inputShape.size() == 3);

    auto const windows = windowIndices();

    Values out(getOutputSize(), Float{});
    for (std::size_t outIdx = 0; outIdx < out.size(); ++outIdx)
    {
        Float best = -std::numeric_limits<Float>::infinity();
        for (std::size_t inIdx : windows[outIdx])
            best = std::max(best, input[inIdx]);
        out[outIdx] = best;
    }

    return out;
}

} // namespace spexplain

