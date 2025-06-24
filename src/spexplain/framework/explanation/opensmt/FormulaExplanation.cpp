#include "FormulaExplanation.h"

#include <spexplain/framework/expand/Expand.h>
#include <spexplain/framework/expand/strategy/Strategy.h>

#include <spexplain/common/Macro.h>

#include <verifiers/opensmt/OpenSMTVerifier.h>

#include <api/MainSolver.h>

#include <cassert>
#include <ostream>

namespace spexplain::opensmt {
FormulaExplanation::FormulaExplanation(Framework const & fw, Formula const & phi)
    : Explanation{fw},
      formulaPtr{MAKE_UNIQUE(phi)} {}

xai::verifiers::OpenSMTVerifier const & FormulaExplanation::getVerifier() const {
    assert(dynamic_cast<xai::verifiers::OpenSMTVerifier const *>(&getExpand().getVerifier()));
    return static_cast<xai::verifiers::OpenSMTVerifier const &>(getExpand().getVerifier());
}

bool FormulaExplanation::contains(VarIdx idx) const {
    auto & verifier = getVerifier();
    return verifier.contains(*formulaPtr, idx);
}

std::size_t FormulaExplanation::termSize() const {
    auto & verifier = getVerifier();
    return verifier.termSizeOf(*formulaPtr);
}

void FormulaExplanation::clear() {
    Explanation::clear();

    resetFormula();
}

void FormulaExplanation::resetFormula() {
    formulaPtr.reset();
}

void FormulaExplanation::swap(FormulaExplanation & rhs) {
    Explanation::swap(rhs);

    formulaPtr.swap(rhs.formulaPtr);
}

void FormulaExplanation::intersect(FormulaExplanation && rhs) {
    auto & solver = getVerifier().getSolver();
    auto & logic = solver.getLogic();

    auto & phi = *formulaPtr;
    auto & phi2 = rhs.getFormula();
    Formula newPhi = logic.mkAnd(phi, phi2);

    phi = std::move(newPhi);
}

void FormulaExplanation::printSmtLib2(std::ostream & os) const {
    auto & solver = getVerifier().getSolver();
    auto & logic = solver.getLogic();
    os << logic.printTerm(*formulaPtr);
}
} // namespace spexplain::opensmt
