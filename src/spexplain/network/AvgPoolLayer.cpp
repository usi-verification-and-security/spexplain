#include "AvgPoolLayer.h"

#include <cassert>

namespace spexplain {

AvgPoolLayer::Shape AvgPoolLayer::makeOutputShape(Shape const & inputShape,
                                                  Kernel const & kernel,
                                                  Strides const & strides,
                                                  Padding const & padding)
{
    // inputShape = {C, H, W}
    std::size_t outH = (inputShape[1] + padding[0] + padding[1] - kernel[0]) / strides[0] + 1;
    std::size_t outW = (inputShape[2] + padding[2] + padding[3] - kernel[1]) / strides[1] + 1;
    return {inputShape[0], outH, outW};
}

AvgPoolLayer::AvgPoolLayer(Shape inputShape, Kernel kernel, Strides strides, Padding padding, bool countIncludePad)
    : NetworkLayer{"avgpool", inputShape, makeOutputShape(inputShape, kernel, strides, padding)},
      kernel{kernel}, strides{strides}, padding{padding}, countIncludePad{countIncludePad}
{
}

std::vector<std::vector<std::size_t>> AvgPoolLayer::windowIndices() const
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

std::vector<std::size_t> AvgPoolLayer::windowDivisors() const
{
    auto const windows = windowIndices();

    std::vector<std::size_t> divisors(windows.size());
    std::size_t const kernelArea = kernel[0] * kernel[1];
    for (std::size_t i = 0; i < windows.size(); ++i)
    {
        // count_include_pad = 1 divides by the full kernel area, treating padding cells as zeros;
        // count_include_pad = 0 (the ONNX default) divides by the number of real input cells.
        divisors[i] = countIncludePad ? kernelArea : windows[i].size();
    }

    return divisors;
}

NetworkLayer::Values AvgPoolLayer::computeLayerOutput(Values const & input) const
{
    assert(input.size() == getInputSize());
    assert(inputShape.size() == 3);

    auto const windows = windowIndices();
    auto const divisors = windowDivisors();

    Values out(getOutputSize(), Float{});
    for (std::size_t outIdx = 0; outIdx < out.size(); ++outIdx)
    {
        Float sum{};
        for (std::size_t inIdx : windows[outIdx])
            sum += input[inIdx];
        assert(divisors[outIdx] > 0);
        out[outIdx] = sum / static_cast<Float>(divisors[outIdx]);
    }

    return out;
}

} // namespace spexplain
