#include "SigmoidLayer.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace spexplain {

NetworkLayer::Values SigmoidLayer::computeLayerOutput(Values const & input) const
{
    assert(input.size() == getInputSize());
    Values output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](Float x) {
        return Float{1} / (Float{1} + std::exp(-x));
    });
    return output;
}

} // namespace spexplain

