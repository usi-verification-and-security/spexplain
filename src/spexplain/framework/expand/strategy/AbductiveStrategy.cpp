#include "AbductiveStrategy.h"

#include <spexplain/framework/explanation/IntervalExplanation.h>
#include <spexplain/framework/explanation/VarBound.h>

#include <verifiers/Verifier.h>

#include <cassert>

namespace spexplain {
void Framework::Expand::AbductiveStrategy::executeBody(Explanations & explanations, Network::Dataset const &,
                                                       ExplanationIdx idx) {
    auto & explanation = getExplanation(explanations, idx);
    assert(dynamic_cast<IntervalExplanation *>(&explanation));
    auto & iexplanation = static_cast<IntervalExplanation &>(explanation);

    auto & verifier = getVerifier();

    for (VarIdx idxToOmit : varOrdering.order) {
        verifier.push();
        assertIntervalExplanationExcept(iexplanation, idxToOmit, {.ignoreVarOrder = true});
        bool const ok = checkFormsExplanation();
        verifier.pop();
        // It is no longer explanation after the removal -> we cannot remove it
        if (not ok) { continue; }
        iexplanation.eraseVarBound(idxToOmit);
    }
}
} // namespace spexplain
