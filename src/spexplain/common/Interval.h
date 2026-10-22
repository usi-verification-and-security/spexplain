#ifndef SPEXPLAIN_INTERVAL_H
#define SPEXPLAIN_INTERVAL_H

#include "Core.h"

#include <cassert>
#include <iosfwd>

namespace spexplain {
class Interval {
public:
    struct Bounds {
        Float lower;
        Float upper;
    };

    constexpr Interval(Bounds bnd) : bounds{std::move(bnd)} { assert(isValid()); }
    constexpr Interval(Float lo, Float hi) : Interval(Bounds{lo, hi}) {}
    constexpr Interval(Float val) : Interval(val, val) {}

    constexpr bool isPoint() const { return size() == 0; }

    constexpr Bounds const & getBounds() const { return bounds; }

    constexpr Float getLower() const { return bounds.lower; }
    constexpr Float getUpper() const { return bounds.upper; }

    constexpr void setLower(Float val) {
        bounds.lower = val;
        assert(isValid());
    }
    constexpr void setUpper(Float val) {
        bounds.upper = val;
        assert(isValid());
    }
    constexpr void incLower(Float val) {
        bounds.lower += val;
        assert(isValid());
    }
    constexpr void incUpper(Float val) {
        bounds.upper += val;
        assert(isValid());
    }
    constexpr void mulLower(Float val) {
        bounds.lower *= val;
        assert(isValid());
    }
    constexpr void mulUpper(Float val) {
        bounds.upper *= val;
        assert(isValid());
    }

    Float getValue() const { return getLower(); }

    constexpr Float size() const { return getUpper() - getLower(); }

    // It assumes that they have at least some overlap
    void intersect(Interval &&);

    void print(std::ostream &) const;

protected:
    constexpr bool isValid() const { return size() >= 0; }

    Bounds bounds;
};
} // namespace spexplain

#endif // SPEXPLAIN_INTERVAL_H
