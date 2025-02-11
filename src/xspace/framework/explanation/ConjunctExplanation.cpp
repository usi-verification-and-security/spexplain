#include "ConjunctExplanation.h"

#include <xspace/common/Print.h>

#include <algorithm>
#include <cassert>

namespace xspace {
ConjunctExplanation::Conjunction::value_type const & ConjunctExplanation::operator[](std::size_t idx) const {
    assert(idx < size());
    return conjunction[idx];
}

ConjunctExplanation::Conjunction::value_type & ConjunctExplanation::operator[](std::size_t idx) {
    return const_cast<ConjunctExplanation::Conjunction::value_type &>(std::as_const(*this)[idx]);
}

bool ConjunctExplanation::contains(VarIdx idx) const {
    return std::ranges::any_of(conjunction, [idx](auto & pexplanationPtr) {
        if (not pexplanationPtr) { return false; }
        return pexplanationPtr->contains(idx);
    });
}

std::size_t ConjunctExplanation::termSize() const {
    std::size_t size_{};
    for (auto & pexplanationPtr : conjunction) {
        if (not pexplanationPtr) { continue; }
        size_ += pexplanationPtr->termSize();
    }

    assert(size_ > 0);
    return size_;
}

void ConjunctExplanation::clear() {
    Explanation::clear();

    for (auto & expPtr : conjunction) {
        expPtr.reset();
    }
}

void ConjunctExplanation::swap(ConjunctExplanation & rhs) {
    Explanation::swap(rhs);

    conjunction.swap(rhs.conjunction);
}

void ConjunctExplanation::insertExplanation(std::unique_ptr<PartialExplanation> pexplanationPtr) {
    assert(pexplanationPtr);
    conjunction.push_back(std::move(pexplanationPtr));
}

bool ConjunctExplanation::eraseExplanation(std::size_t idx) {
    auto & pexplanationPtr = operator[](idx);
    return eraseExplanation(pexplanationPtr);
}

bool ConjunctExplanation::eraseExplanation(Conjunction::iterator it) {
    std::unique_ptr<PartialExplanation> & pexplanationPtr = *it;
    return eraseExplanation(pexplanationPtr);
}

bool ConjunctExplanation::eraseExplanation(std::unique_ptr<PartialExplanation> & pexplanationPtr) {
    if (not pexplanationPtr) { return false; }
    pexplanationPtr.reset();
    return true;
}

void ConjunctExplanation::merge(ConjunctExplanation && cexplanation) {
    for (auto & pexplanationPtr : cexplanation.conjunction) {
        if (not pexplanationPtr) { continue; }
        insertExplanation(std::move(pexplanationPtr));
    }
}

void ConjunctExplanation::printSmtLib2(std::ostream & os) const {
    printSmtLib2(os, PrintConfig{});
}

void ConjunctExplanation::printSmtLib2(std::ostream & os, PrintConfig const & conf) const {
    os << "(and";
    for (auto & pexplanationPtr : conjunction) {
        if (not pexplanationPtr) { continue; }
        auto & explanation = *pexplanationPtr;
        os << conf.delim << smtLib2Format(explanation);
    }
    os << ')';
}
} // namespace xspace
