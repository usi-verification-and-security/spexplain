#include "AddLayer.h"

#include <cassert>

namespace spexplain {

NetworkLayer::Values AddLayer::computeLayerOutput(Values const & input) const
{
    assert(input.size() == getInputSize());

    Values out(input);

    if (bias.empty())
        return out;

    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] += biasFor(i);

    return out;
}

Float AddLayer::biasFor(std::size_t idx) const
{
    if (bias.empty())
        return Float{};

    if (bias.size() == 1)
        return bias[0];

    std::size_t const outSize = getOutputSize();
    if (bias.size() == outSize)
        return bias[idx];

    // Common FC case: bias matches trailing dimension and broadcasts over leading dims.
    if (!outputShape.empty() && bias.size() == outputShape.back())
        return bias[idx % outputShape.back()];

    // Fallback: repeat cyclically.
    return bias[idx % bias.size()];
}

} // namespace spexplain

