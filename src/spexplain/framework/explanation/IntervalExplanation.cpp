#include "IntervalExplanation.h"

#include "../Config.h"

#include <spexplain/common/Macro.h>
#include <spexplain/common/Print.h>

#include <algorithm>
#include <ostream>

namespace spexplain {
IntervalExplanation::IntervalExplanation(Framework const & fw) : ConjunctExplanation{fw, fw.varSize()} {
    assert(size() == frameworkPtr->varSize());
}

std::size_t IntervalExplanation::varSize() const {
    return validSize();
}

void IntervalExplanation::insertExplanation(std::unique_ptr<PartialExplanation>) {
    // Not compatible
    assert(false);
}

void IntervalExplanation::insertVarBound(VarBound varBnd) {
    VarIdx idx = varBnd.getVarIdx();
    assert(not contains(idx));
    auto & pexplanationPtr = operator[](idx);
    auto varBndPtr = MAKE_UNIQUE(std::move(varBnd));
    pexplanationPtr = std::move(varBndPtr);
}

void IntervalExplanation::insertBound(VarIdx idx, Bound bnd) {
#ifndef NDEBUG
    auto & network = frameworkPtr->getNetwork();
    Float domainLower = network.getInputLowerBound(idx);
    Float domainUpper = network.getInputUpperBound(idx);
    Float val = bnd.getValue();
    bool const valIsLower = (val == domainLower);
    bool const valIsUpper = (val == domainUpper);
    assert(not valIsLower or not valIsUpper);
    // It is worthless to assert bounds that already correspond to the bounds of the domain
    assert(not valIsLower or bnd.isEq());
    assert(not valIsUpper or bnd.isEq());
#endif

    auto * optVarBnd = tryGetVarBound(idx);
    if (not optVarBnd) {
        insertVarBound(VarBound{*frameworkPtr, idx, std::move(bnd)});
        return;
    }

    VarBound & varBnd = *optVarBnd;
    assert(not varBnd.isInterval());
    assert(not varBnd.isPoint());
    assert(not bnd.isEq());
    assert(bnd.isLower() xor varBnd.getBound().isLower());
    assert(bnd.isUpper() xor varBnd.getBound().isUpper());
    varBnd.insertBound(std::move(bnd));
}

void IntervalExplanation::condense() {
    // The sparseness needs to be kept here ...
    assert(false);
}

void IntervalExplanation::intersect(std::unique_ptr<Explanation> && explanationPtr) {
    if (auto optIntExp = dynamic_cast<IntervalExplanation *>(explanationPtr.get())) {
        intersect(std::move(*optIntExp));
        return;
    }

    ConjunctExplanation::intersect(std::move(explanationPtr));
}

void IntervalExplanation::intersect(ConjunctExplanation &&) {
    // Would result in non-IntervalExplanation
    assert(false);
}

void IntervalExplanation::intersect(IntervalExplanation && rhs) {
    assert(size() == rhs.size());

    std::size_t const size_ = size();
    for (VarIdx idx = 0; idx < size_; ++idx) {
        auto & varBndPtr2 = rhs[idx];
        if (not varBndPtr2) { continue; }

        auto & varBndPtr = operator[](idx);
        if (not varBndPtr) {
            varBndPtr = std::move(varBndPtr2);
            continue;
        }

        assert(varBndPtr and varBndPtr2);
        auto & varBnd = *castToVarBound(varBndPtr.get());
        auto & varBnd2 = *castToVarBound(varBndPtr2.get());
        // It assumes that they have at least some overlap
        varBnd.intersect(std::move(varBnd2));
    }
}

std::unique_ptr<ConjunctExplanation>
IntervalExplanation::toConjunctExplanation(std::vector<VarIdx> const & varOrder) && {
    [[maybe_unused]] auto const varSize_ = varSize();

    ConjunctExplanation cexplanation{*frameworkPtr};

    for (VarIdx idx : varOrder) {
        auto & varBndPtr = operator[](idx);
        if (not varBndPtr) { continue; }
        cexplanation.insertExplanation(std::move(varBndPtr));
    }

    assert(cexplanation.size() == varSize_);
    assert(not cexplanation.isSparse());

    return MAKE_UNIQUE(std::move(cexplanation));
}

std::size_t IntervalExplanation::computeFixedCount() const {
    return std::ranges::count_if(*this, [](auto const & expPtr) {
        if (not expPtr) { return false; }
        auto & varBnd = *castToVarBound(expPtr.get());
        return varBnd.isPoint();
    });
}

Float IntervalExplanation::getRelativeVolume() const {
    return computeRelativeVolumeTp<false>();
}

Float IntervalExplanation::getRelativeVolumeSkipFixed() const {
    return computeRelativeVolumeTp<true>();
}

template<bool skipFixed>
Float IntervalExplanation::computeRelativeVolumeTp() const {
    Float relVolume = 1;
    for (auto const & expPtr : *this) {
        if (not expPtr) { continue; }
        auto & varBnd = *castToVarBound(expPtr.get());
        Float const size = varBnd.size();
        assert(size >= 0);
        if (size == 0) {
            if constexpr (skipFixed) {
                continue;
            } else {
                return 0;
            }
        }

        auto const & idx = varBnd.getVarIdx();
        Float const domainSize = frameworkPtr->getDomainInterval(idx).size();
        assert(domainSize > 0);
        assert(size < domainSize);

        relVolume *= size / domainSize;
    }

    assert(relVolume > 0);
    assert(relVolume <= 1);
    return relVolume;
}

IntervalExplanation::PrintFormat const & IntervalExplanation::getPrintFormat() const {
    return frameworkPtr->getConfig().getPrintingIntervalExplanationsFormat();
}

void IntervalExplanation::print(std::ostream & os) const {
    PrintFormat const & type = getPrintFormat();
    using enum PrintFormat;
    switch (type) {
        case smtlib2:
            printSmtLib2(os);
            return;
        case bounds:
            printBounds(os);
            return;
        case intervals:
            printIntervals(os);
            return;
    }
}

void IntervalExplanation::print(std::ostream & os, PrintConfig const & conf) const {
    PrintFormat const & type = getPrintFormat();
    using enum PrintFormat;
    switch (type) {
        case smtlib2:
            printSmtLib2(os, conf);
            return;
        case bounds:
            printBounds(os, conf);
            return;
        case intervals:
            printIntervals(os, conf);
            return;
    }
}

void IntervalExplanation::printBounds(std::ostream & os, PrintConfig const & conf) const {
    printTp<PrintFormat::bounds>(os, conf);
}

void IntervalExplanation::printIntervals(std::ostream & os, PrintConfig const & conf) const {
    printTp<PrintFormat::intervals>(os, conf);
}

template<IntervalExplanation::PrintFormat type>
void IntervalExplanation::printTp(std::ostream & os, PrintConfig const & conf) const {
    constexpr bool isSmtLib2 = (type == PrintFormat::smtlib2);
    constexpr bool isBounds = (type == PrintFormat::bounds);
    constexpr bool isIntervals = (type == PrintFormat::intervals);
    static_assert(not isSmtLib2);

    bool const includeAll = conf.includeAll;

    auto const size_ = size();
    for (VarIdx idx = 0; idx < size_; ++idx) {
        auto const * optVarBnd = tryGetVarBound(idx);
        if (includeAll) {
            if constexpr (isBounds) {
                printElemBounds(os, conf, idx, optVarBnd);
            } else {
                static_assert(isIntervals);
                printElemInterval(os, conf, idx, optVarBnd);
            }
            continue;
        }

        if (not optVarBnd) { continue; }
        auto & varBnd = *optVarBnd;
        if constexpr (isBounds) {
            printElemBounds(os, conf, varBnd);
        } else {
            static_assert(isIntervals);
            printElemInterval(os, conf, varBnd);
        }
    }
}

void IntervalExplanation::printElemBounds(std::ostream & os, PrintConfig const & conf, VarBound const & varBnd) {
    os << varBnd << conf.delim;
}

void IntervalExplanation::printElemInterval(std::ostream & os, PrintConfig const & conf,
                                            VarBound const & varBnd) const {
    os << varBnd.toInterval() << conf.delim;
}

void IntervalExplanation::printElemBounds(std::ostream & os, PrintConfig const & conf, VarIdx idx,
                                          VarBound const * optVarBnd) const {
    if (optVarBnd) {
        printElemBounds(os, conf, *optVarBnd);
    } else {
        os << frameworkPtr->getVarName(idx) << " free" << conf.delim;
    }
}

void IntervalExplanation::printElemInterval(std::ostream & os, PrintConfig const & conf, VarIdx idx,
                                            VarBound const * optVarBnd) const {
    if (optVarBnd) {
        printElemInterval(os, conf, *optVarBnd);
    } else {
        os << frameworkPtr->getDomainInterval(idx) << conf.delim;
    }
}
} // namespace spexplain
