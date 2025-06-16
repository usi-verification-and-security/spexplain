#include "Utils.h"

namespace spexplain {
std::unique_ptr<VarBound> intervalToOptVarBound(Framework const & framework, VarIdx idx, Interval const & ival) {
    auto const [lo, hi] = ival.getBounds();
    auto & network = framework.getNetwork();
    bool const isLower = (lo == network.getInputLowerBound(idx));
    bool const isUpper = (hi == network.getInputUpperBound(idx));
    if (isLower and isUpper) { return nullptr; }

    return std::make_unique<VarBound>(framework, idx, ival);
}
} // namespace spexplain
