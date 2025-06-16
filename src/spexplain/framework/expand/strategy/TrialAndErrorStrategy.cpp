#include "TrialAndErrorStrategy.h"

#include <spexplain/framework/Utils.h>
#include <spexplain/framework/explanation/IntervalExplanation.h>
#include <spexplain/framework/explanation/VarBound.h>

#include <verifiers/Verifier.h>

#include <cassert>

namespace spexplain {
void Framework::Expand::TrialAndErrorStrategy::executeBody(Explanations & explanations, Network::Dataset const &,
                                                           ExplanationIdx idx) {
    auto & explanation = getExplanation(explanations, idx);
    assert(dynamic_cast<IntervalExplanation *>(&explanation));
    auto & iexplanation = static_cast<IntervalExplanation &>(explanation);

    auto & fw = expand.getFramework();
    auto & verifier = getVerifier();
    auto const maxAttempts = config.maxAttempts;
    assert(maxAttempts > 0);

    for (VarIdx idxToRelax : varOrdering.order) {
        auto * optVarBndToRelax = iexplanation.tryGetVarBound(idxToRelax);
        if (not optVarBndToRelax) { continue; }

        verifier.push();
        assertIntervalExplanationExcept(iexplanation, idxToRelax, {.ignoreVarOrder = true});

        auto & varBndToRelax = *optVarBndToRelax;
        Interval origInterval = varBndToRelax.toInterval();
        Interval const & domainInterval = fw.getDomainInterval(idxToRelax);
        auto [oLo, oHi] = origInterval.getBounds();
        auto const [dLo, dHi] = domainInterval.getBounds();
        assert(dLo <= oLo and oHi <= dHi);
        assert(dLo < oLo or oHi < dHi);

        if (oLo != dLo) {
            verifier.push();
            Interval relaxedLowerIval{dLo, oHi};
            for (int i = 0; i < maxAttempts; ++i) {
                Float const lo = relaxedLowerIval.getLower();
                assert(lo < oLo);
                assertInterval(idxToRelax, relaxedLowerIval);
                bool const ok = checkFormsExplanation();
                if (ok) {
                    oLo = lo;
                    origInterval.setLower(oLo);
                    break;
                }
                relaxedLowerIval.setLower((lo + oLo) / 2);
            }
            verifier.pop();
        }

        if (oHi != dHi) {
            verifier.push();
            Interval relaxedUpperIval{oLo, dHi};
            for (int i = 0; i < maxAttempts; ++i) {
                Float const hi = relaxedUpperIval.getUpper();
                assert(hi > oHi);
                assertInterval(idxToRelax, relaxedUpperIval);
                bool const ok = checkFormsExplanation();
                if (ok) {
                    oHi = hi;
                    origInterval.setUpper(oHi);
                    break;
                }
                relaxedUpperIval.setUpper((hi + oHi) / 2);
            }
            verifier.pop();
        }

        verifier.pop();

        auto varBndPtr = intervalToOptVarBound(fw, idxToRelax, std::move(origInterval));
        iexplanation[idxToRelax] = std::move(varBndPtr);
    }
}
} // namespace spexplain
