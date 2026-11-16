#include "Explanation.h"

#include <utility>

namespace spexplain {
void Explanation::swap(Explanation & rhs) {
    PartialExplanation::swap(rhs);

    std::swap(verified, rhs.verified);
}
} // namespace spexplain
