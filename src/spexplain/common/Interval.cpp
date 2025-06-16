#include "Interval.h"

#include <ostream>

namespace spexplain {
void Interval::intersect(Interval && rhs) {
    assert(isValid());
    assert(rhs.isValid());

    setLower(std::max(getLower(), rhs.getLower()));
    setUpper(std::min(getUpper(), rhs.getUpper()));
}

void Interval::print(std::ostream & os) const {
    os << '[' << getLower() << ',' << getUpper() << ']';
}
} // namespace spexplain
