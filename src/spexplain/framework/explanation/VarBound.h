#ifndef SPEXPLAIN_VARBOUND_H
#define SPEXPLAIN_VARBOUND_H

#include "PartialExplanation.h"

#include <spexplain/common/Bound.h>
#include <spexplain/common/Interval.h>
#include <spexplain/common/Var.h>

#include <cassert>
#include <iosfwd>
#include <optional>
#include <type_traits>
#include <variant>

namespace spexplain {
class Framework;

class VarBound : public PartialExplanation {
public:
    explicit VarBound(Framework const & fw, VarIdx idx, Float val) : VarBound(fw, idx, EqBound{val}) {}
    explicit VarBound(Framework const & fw, VarIdx idx, Interval const & ival)
        : VarBound(fw, idx, LowerBound{ival.getLower()}, UpperBound{ival.getUpper()}) {}
    explicit VarBound(Framework const &, VarIdx, EqBound);
    explicit VarBound(Framework const &, VarIdx, LowerBound);
    explicit VarBound(Framework const &, VarIdx, UpperBound);
    explicit VarBound(Framework const &, VarIdx, Bound);
    explicit VarBound(Framework const &, VarIdx, LowerBound, UpperBound);
    explicit VarBound(Framework const &, VarIdx, Bound, Bound);

    bool isInterval() const {
        assert(not isIntervalImpl() or firstBound.isLower());
        assert(not isIntervalImpl() or optSecondBound->isUpper());
        assert(not isIntervalImpl() or firstBound.getValue() < optSecondBound->getValue());
        return isIntervalImpl();
    }

    bool isPoint() const {
        assert(not isPointImpl() or not isInterval());
        return isPointImpl();
    }

    VarIdx getVarIdx() const { return varIdx; }

    bool contains(VarIdx idx) const override { return idx == getVarIdx(); }

    std::size_t varSize() const override { return 1; }
    std::size_t termSize() const override;

    // If it holds just a single LowerBound or UpperBound, it must be accessed from here
    Bound const & getBound() const { return firstBound; }

    LowerBound const & getIntervalLower() const {
        assert(isInterval());
        auto & bnd = getBound();
        assert(bnd.isLower());
        return static_cast<LowerBound const &>(bnd);
    }

    UpperBound const & getIntervalUpper() const {
        assert(isInterval());
        auto & bnd = *optSecondBound;
        assert(bnd.isUpper());
        return static_cast<UpperBound const &>(bnd);
    }

    // Recommended but not necessary to prefer over getBound
    EqBound const & getPoint() const {
        assert(isPoint());
        return static_cast<EqBound const &>(getBound());
    }

    void swap(VarBound &);

    Float size() const { return toInterval().size(); }

    Interval getInterval() const;
    std::optional<Interval> tryGetIntervalOrPoint() const;

    Interval toInterval() const;

    void insertBound(Bound);
    void insertLowerBound(LowerBound);
    void insertUpperBound(UpperBound);

    void eraseLowerBound();
    void eraseUpperBound();

    // It assumes that they have at least some overlap
    void intersect(VarBound &&);

    void print(std::ostream & os) const override { printRegular(os); }
    void printSmtLib2(std::ostream &) const override;

protected:
    void assertValid() const;

    void printRegular(std::ostream &) const;
    void printRegularBound(std::ostream &, Bound const &) const;

    void printSmtLib2Bound(std::ostream &, Bound const &) const;

    VarIdx varIdx;

    //+ it should be more efficient to either store `Interval` or a single `Bound`
    Bound firstBound;
    std::optional<Bound> optSecondBound{};

private:
    struct ConsOneBoundTag {};
    struct ConsPointTag {};
    struct ConsIntervalTag {};

    explicit VarBound(Framework const &, VarIdx, Bound &&, ConsOneBoundTag);
    explicit VarBound(Framework const &, VarIdx, LowerBound &&, UpperBound &&, ConsPointTag);
    explicit VarBound(Framework const &, VarIdx, LowerBound &&, UpperBound &&, ConsIntervalTag);

    bool isIntervalImpl() const { return optSecondBound.has_value(); }

    bool isPointImpl() const { return getBound().isEq(); }
};
} // namespace spexplain

#endif // SPEXPLAIN_VARBOUND_H
