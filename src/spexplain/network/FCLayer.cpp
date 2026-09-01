//
// Created by labbaf on 27.08.2025.
//

#include "FCLayer.h"

#include <cassert>

namespace spexplain {
NetworkLayer::Values FCLayer::computeLayerOutput(Values const & input) const {
    assert(getInputSize() == input.size());
    auto const inputSize = getInputSize();
    auto const outputSize = getOutputSize();
    assert(weights.size() == inputSize * outputSize);

    Values output(outputSize, Float{});
    for (std::size_t out = 0; out < outputSize; ++out) {
        Float sum = biases.empty() ? Float{} : biases[out % biases.size()];
        for (std::size_t in = 0; in < inputSize; ++in) {
            sum += input[in] * weights[in * outputSize + out];
        }
        output[out] = sum;
    }

    return output;
}
} // namespace spexplain
