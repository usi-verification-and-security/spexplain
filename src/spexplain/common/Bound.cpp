#include "Bound.h"

#include <ostream>

namespace spexplain {
void Bound::printRegular(std::ostream & os) const {
    std::visit([&](auto & t) { os << t.getSymbol() << ' ' << value; }, op);
}

void Bound::printReverse(std::ostream & os) const {
    std::visit([&](auto & t) { os << value << ' ' << t.getReverseSymbol(); }, op);
}
} // namespace spexplain
