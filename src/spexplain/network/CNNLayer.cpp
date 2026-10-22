#include "CNNLayer.h"

#include <cassert>
#include <stdexcept>

namespace spexplain {

// ---------------------------------------------------------------------------
// Helper – compute output spatial size for one dimension
// ---------------------------------------------------------------------------
static std::size_t outputDim(std::size_t inputDim, std::size_t filterDim,
                              std::size_t stride, std::size_t padBefore,
                              std::size_t padAfter)
{
    std::size_t padded = inputDim + padBefore + padAfter;
    if (padded < filterDim)
        throw std::invalid_argument("CNNLayer: filter larger than padded input dimension");
    return (padded - filterDim) / stride + 1;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CNNLayer::CNNLayer(Shape inputShape,
                   Shape filterShape,
                   Values filters,
                   Values biases,
                   Strides strides,
                   Padding padding)
    : NetworkLayer{"cnn",
                   inputShape,
                   // outputShape = {outC, outH, outW}
                   {filterShape[0],
                    outputDim(inputShape[1], filterShape[2], strides[0], padding[0], padding[1]),
                    outputDim(inputShape[2], filterShape[3], strides[1], padding[2], padding[3])}},
      filterShape{std::move(filterShape)},
      filters{std::move(filters)},
      biases{std::move(biases)},
      strides{strides},
      padding{padding}
{
    assert(this->filterShape.size() == 4);   // {outC, inC, kH, kW}
    assert(this->inputShape.size()  == 3);   // {inC, H, W}
    assert(this->outputShape.size() == 3);   // {outC, outH, outW}
}

// ---------------------------------------------------------------------------
// Forward pass: standard 2-D cross-correlation
// ---------------------------------------------------------------------------
NetworkLayer::Values CNNLayer::computeLayerOutput(Values const & input) const {
    assert(input.size() == getInputSize());

    std::size_t const inC   = inputShape[0];
    std::size_t const inH   = inputShape[1];
    std::size_t const inW   = inputShape[2];

    std::size_t const outC  = filterShape[0];
    std::size_t const kH    = filterShape[2];
    std::size_t const kW    = filterShape[3];

    std::size_t const outH  = outputShape[1];
    std::size_t const outW  = outputShape[2];

    std::size_t const sH    = strides[0];
    std::size_t const sW    = strides[1];
    std::size_t const padT  = padding[0];   // top
    std::size_t const padL  = padding[2];   // left

    Values output(outC * outH * outW, Float{});

    for (std::size_t oc = 0; oc < outC; ++oc) {
        Float bias = biases.empty() ? Float{} : biases[oc];
        for (std::size_t oh = 0; oh < outH; ++oh) {
            for (std::size_t ow = 0; ow < outW; ++ow) {
                Float sum = bias;

                for (std::size_t ic = 0; ic < inC; ++ic) {
                    for (std::size_t kh = 0; kh < kH; ++kh) {
                        for (std::size_t kw = 0; kw < kW; ++kw) {
                            // Input spatial indices (with padding offset)
                            std::size_t const ih = oh * sH + kh;
                            std::size_t const iw = ow * sW + kw;

                            // Skip if this position is inside the padding region
                            if (ih < padT || iw < padL) continue;
                            std::size_t const realH = ih - padT;
                            std::size_t const realW = iw - padL;
                            if (realH >= inH || realW >= inW) continue;

                            Float const in_val =
                                input[ic * inH * inW + realH * inW + realW];
                            Float const w_val =
                                filters[oc * inC * kH * kW + ic * kH * kW + kh * kW + kw];
                            sum += in_val * w_val;
                        }
                    }
                }

                output[oc * outH * outW + oh * outW + ow] = sum;
            }
        }
    }

    return output;
}

} // namespace spexplain

