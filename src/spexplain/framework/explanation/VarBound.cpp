#include "VarBound.h"

#include "../Print.h"

#include <ostream>

namespace spexplain {
VarBound::VarBound(Framework const & fw, VarIdx idx, EqBound eq) : VarBound(fw, idx, std::move(eq), ConsOneBoundTag{}) {
    assert(isPoint());
}

VarBound::VarBound(Framework const & fw, VarIdx idx, LowerBound lo)
    : VarBound(lo.getValue() == fw.getNetwork().getInputUpperBound(idx)
                   ? VarBound{fw, idx, lo.getValue()}
                   : VarBound{fw, idx, std::move(lo), ConsOneBoundTag{}}) {}

VarBound::VarBound(Framework const & fw, VarIdx idx, UpperBound hi)
    : VarBound(hi.getValue() == fw.getNetwork().getInputLowerBound(idx)
                   ? VarBound{fw, idx, hi.getValue()}
                   : VarBound{fw, idx, std::move(hi), ConsOneBoundTag{}}) {}

VarBound::VarBound(Framework const & fw, VarIdx idx, Bound bnd)
    : VarBound(bnd.isEq()      ? VarBound{fw, idx, static_cast<EqBound &&>(bnd)}
               : bnd.isLower() ? VarBound{fw, idx, static_cast<LowerBound &&>(bnd)}
                               : VarBound{fw, idx, static_cast<UpperBound &&>(bnd)}) {}

VarBound::VarBound(Framework const & fw, VarIdx idx, LowerBound lo, UpperBound hi)
    : VarBound(lo.getValue() == hi.getValue() ? VarBound{fw, idx, std::move(lo), std::move(hi), ConsPointTag{}}
                                              : VarBound{fw, idx, std::move(lo), std::move(hi), ConsIntervalTag{}}) {}

VarBound::VarBound(Framework const & fw, VarIdx idx, Bound bnd1, Bound bnd2)
    : VarBound(bnd1.isLower() ? VarBound{fw, idx, static_cast<LowerBound &&>(bnd1), static_cast<UpperBound &&>(bnd2)}
                              : VarBound{fw, idx, static_cast<LowerBound &&>(bnd2), static_cast<UpperBound &&>(bnd1)}) {
}

VarBound::VarBound(Framework const & fw, VarIdx idx, Bound && bnd, ConsOneBoundTag)
    : PartialExplanation{fw},
      varIdx{idx},
      firstBound{std::move(bnd)} {
    assert(not isInterval());
    assertValid();
}

VarBound::VarBound(Framework const & fw, VarIdx idx, LowerBound && lo, [[maybe_unused]] UpperBound && hi, ConsPointTag)
    : VarBound(fw, idx, lo.getValue()) {
    assert(lo.isLower());
    assert(hi.isUpper());
    assert(lo.getValue() == hi.getValue());
}

VarBound::VarBound(Framework const & fw, VarIdx idx, LowerBound && lo, UpperBound && hi, ConsIntervalTag)
    : PartialExplanation{fw},
      varIdx{idx},
      firstBound{std::move(lo)},
      optSecondBound{std::move(hi)} {
    assert(firstBound.isLower());
    assert(optSecondBound->isUpper());
    assert(isInterval());

    auto & network = fw.getNetwork();
    bool const isLower = (firstBound.getValue() == network.getInputLowerBound(idx));
    bool const isUpper = (optSecondBound->getValue() == network.getInputUpperBound(idx));
    assert(not isLower or not isUpper);
    if (isLower or isUpper) {
        assert(isLower xor isUpper);
        if (isLower) {
            eraseLowerBound();
        } else {
            eraseUpperBound();
        }
        assert(not isInterval());
        assert(not isPoint());
    }

    assertValid();
}

void VarBound::assertValid() const {
#ifndef NDEBUG
    auto const [dLo, dHi] = frameworkPtr->getDomainInterval(varIdx).getBounds();
    if (isInterval()) {
        auto ival = getInterval();
        auto const [lo, hi] = ival.getBounds();
        assert(lo <= hi);
        assert(lo > dLo and hi < dHi);
        return;
    }
    auto & bnd = getBound();
    Float const val = bnd.getValue();
    if (bnd.isEq()) {
        assert(val >= dLo and val <= dHi);
    } else {
        assert(val > dLo and val < dHi);
    }
#endif
}

std::size_t VarBound::termSize() const {
    if (isInterval()) return 2;
    return 1;
}

void VarBound::swap(VarBound & rhs) {
    PartialExplanation::swap(rhs);

    std::swap(firstBound, rhs.firstBound);
    std::swap(optSecondBound, rhs.optSecondBound);
}

Interval VarBound::getInterval() const {
    assert(isInterval());
    return Interval{firstBound.getValue(), optSecondBound->getValue()};
}

std::optional<Interval> VarBound::tryGetIntervalOrPoint() const {
    if (isInterval()) { return getInterval(); }
    if (isPoint()) { return Interval{firstBound.getValue()}; }
    return std::nullopt;
}

Interval VarBound::toInterval() const {
    if (auto optIval = tryGetIntervalOrPoint()) { return *std::move(optIval); }
    assert(not isPoint());
    assert(not isInterval());

    auto & bnd = getBound();
    assert(bnd.isLower() xor bnd.isUpper());
    bool const isLower = bnd.isLower();

    auto & network = frameworkPtr->getNetwork();
    Float val = bnd.getValue();
    auto ival =
        isLower ? Interval{val, network.getInputUpperBound(varIdx)} : Interval{network.getInputLowerBound(varIdx), val};
    assert(not ival.isPoint());
    return ival;
}

void VarBound::insertBound(Bound bnd) {
    assert(not bnd.isEq());
    if (bnd.isLower()) {
        insertLowerBound(static_cast<LowerBound &&>(bnd));
    } else {
        assert(bnd.isUpper());
        insertUpperBound(static_cast<UpperBound &&>(bnd));
    }
}

void VarBound::insertLowerBound(LowerBound lo) {
    assert(not isInterval());
    assert(not isPoint());
    assert(firstBound.isUpper());
    assert(not optSecondBound.has_value());
    Float const val = lo.getValue();
    if (val == firstBound.getValue()) {
        firstBound = EqBound{val};
        assert(isPoint());
    } else {
        optSecondBound.emplace(std::move(firstBound));
        firstBound = std::move(lo);
        assert(isInterval());
    }
    assertValid();
}

void VarBound::insertUpperBound(UpperBound hi) {
    assert(not isInterval());
    assert(not isPoint());
    assert(firstBound.isLower());
    assert(not optSecondBound.has_value());
    Float const val = hi.getValue();
    if (val == firstBound.getValue()) {
        firstBound = EqBound{val};
        assert(isPoint());
    } else {
        optSecondBound.emplace(std::move(hi));
        assert(isInterval());
    }
    assertValid();
}

void VarBound::eraseLowerBound() {
    assert(isInterval());
    firstBound = std::move(*optSecondBound);
    assert(optSecondBound.has_value());
    optSecondBound.reset();
    assert(not isInterval());
    assert(not isPoint());
    assertValid();
}

void VarBound::eraseUpperBound() {
    assert(isInterval());
    optSecondBound.reset();
    assert(not isInterval());
    assert(not isPoint());
    assertValid();
}

void VarBound::intersect(VarBound && rhs) {
#ifndef NDEBUG
    assert(varIdx == rhs.varIdx);
    toInterval().intersect(rhs.toInterval());
#endif

    if (isPoint()) { return; }
    if (rhs.isPoint()) {
        swap(rhs);
        return;
    }

    Interval ival{toInterval()};
    ival.intersect(std::move(rhs).toInterval());
    VarBound varBnd{*frameworkPtr, varIdx, std::move(ival)};
    swap(varBnd);
}

void VarBound::printRegular(std::ostream & os) const {
    if (not isInterval()) {
        printRegularBound(os, firstBound);
        return;
    }

    // x >= l && x <= u -> l <= x <= u
    firstBound.printReverse(os);
    os << ' ';
    printRegularBound(os, *optSecondBound);
}

void VarBound::printRegularBound(std::ostream & os, Bound const & bnd) const {
    os << frameworkPtr->getVarName(varIdx) << ' ' << bnd;
}

void VarBound::printSmtLib2(std::ostream & os) const {
    bool const ival = isInterval();

    if (ival) { os << "(and "; }
    printSmtLib2Bound(os, firstBound);
    if (ival) {
        printSmtLib2Bound(os, *optSecondBound);
        os << ')';
    }
}

void VarBound::printSmtLib2Bound(std::ostream & os, Bound const & bnd) const {
    os << '(' << bnd.getSymbol() << ' ' << frameworkPtr->getVarName(varIdx) << ' ';
    printSmtLib2AsRational(os, bnd.getValue());
    os << ')';
}
} // namespace spexplain
