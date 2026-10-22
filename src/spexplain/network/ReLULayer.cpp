#include "ReLULayer.h"

#include <algorithm>
#include <cassert>

namespace spexplain {

NetworkLayer::Values ReLULayer::computeLayerOutput(Values const & input) const {
    assert(input.size() == getInputSize());
    Values output(input.size());
    std::transform(input.begin(), input.end(), output.begin(),
                   [](Float x) { return x > Float{0} ? x : Float{0}; });
    return output;
}

} // namespace spexplain

