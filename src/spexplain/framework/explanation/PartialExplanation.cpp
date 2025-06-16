#include "PartialExplanation.h"

#include <cassert>
#include <utility>

namespace spexplain {
std::size_t PartialExplanation::varSize() const {
    std::size_t const varSize_ = frameworkPtr->varSize();
    std::size_t cnt{};
    for (VarIdx idx = 0; idx < varSize_; ++idx) {
        if (contains(idx)) { ++cnt; }
    }
    return cnt;
}

void PartialExplanation::swap([[maybe_unused]] PartialExplanation & rhs) {
    assert(frameworkPtr == rhs.frameworkPtr);
}
} // namespace spexplain
