#include "InterpolationStrategy.h"

#include <xspace/framework/explanation/ConjunctExplanation.h>
#include <xspace/framework/explanation/IntervalExplanation.h>
#include <xspace/framework/explanation/opensmt/FormulaExplanation.h>

#include <xspace/common/Utils.h>

#include <verifiers/opensmt/OpenSMTVerifier.h>

#include <api/MainSolver.h>

#include <cassert>

namespace xspace::expand::opensmt {
void InterpolationStrategy::executeInit(std::unique_ptr<Explanation> & explanationPtr) {
    Strategy::executeInit(explanationPtr);

    auto & verifier = getVerifier();
    auto & solver = verifier.getSolver();
    auto & solverConf = solver.getConfig();

    solverConf.setReduction(true);
    solverConf.setSimplifyInterpolant(4);

    switch (config.boolInterpolationAlg) {
        case BoolInterpolationAlg::weak:
            solverConf.setBooleanInterpolationAlgorithm(::opensmt::itp_alg_mcmillanp);
            break;
        case BoolInterpolationAlg::strong:
            solverConf.setBooleanInterpolationAlgorithm(::opensmt::itp_alg_mcmillan);
            break;
    }

    switch (config.arithInterpolationAlg) {
        case ArithInterpolationAlg::weak:
            solverConf.setLRAInterpolationAlgorithm(::opensmt::itp_lra_alg_weak);
            break;
        case ArithInterpolationAlg::weaker:
            solverConf.setLRAInterpolationAlgorithm(::opensmt::itp_lra_alg_decomposing_weak);
            break;
        case ArithInterpolationAlg::strong:
            solverConf.setLRAInterpolationAlgorithm(::opensmt::itp_lra_alg_strong);
            break;
        case ArithInterpolationAlg::stronger:
            solverConf.setLRAInterpolationAlgorithm(::opensmt::itp_lra_alg_decomposing_strong); // seems too strong ...
            break;
    }
}

void InterpolationStrategy::executeBody(std::unique_ptr<Explanation> & explanationPtr) {
    if (auto * optIntExp = dynamic_cast<IntervalExplanation *>(explanationPtr.get())) {
        explanationPtr = std::move(*optIntExp).toConjunctExplanation(varOrdering.order);
    }

    assert(dynamic_cast<ConjunctExplanation *>(explanationPtr.get()));
    auto & cexplanation = static_cast<ConjunctExplanation &>(*explanationPtr);

    auto const & varIndicesFilter = config.varIndicesFilter;
    // Would not work for non-conj. explanations
    bool const filteringVars = not varIndicesFilter.empty();

    if (filteringVars) {
        if (std::ranges::none_of(varIndicesFilter,
                                 [&cexplanation](VarIdx varIdx) { return cexplanation.contains(varIdx); })) {
            return;
        }
    }

    auto & fw = expand.getFramework();
    auto & verifier = getVerifier();
    auto & solver = verifier.getSolver();

    cexplanation.condense();
    auto const firstFormulaIdx = solver.getInsertedFormulasCount();
    // We already took care of possible var ordering in toConjunctExplanation
    assertConjunctExplanation(cexplanation, {.ignoreVarOrder = true});
    auto const lastFormulaIdx = solver.getInsertedFormulasCount() - 1;

    [[maybe_unused]] bool const ok = checkFormsExplanation();
    assert(ok);

    ::opensmt::ipartitions_t part = 0;
    assert(not cexplanation.isSparse());
    for (auto formulaIdx = firstFormulaIdx; formulaIdx <= lastFormulaIdx; ++formulaIdx) {
        std::size_t const expIdx = formulaIdx - firstFormulaIdx;
        if (filteringVars) {
            assert(expIdx < cexplanation.size());
            auto * optExp = cexplanation.tryGetExplanation(expIdx);
            assert(optExp);
            auto & pexplanation = *optExp;
            if (std::ranges::none_of(varIndicesFilter,
                                     [&pexplanation](VarIdx varIdx) { return pexplanation.contains(varIdx); })) {
                continue;
            }
        }

        ::opensmt::setbit(part, formulaIdx);
        if (not filteringVars) { continue; }

        [[maybe_unused]] bool const erased = cexplanation.eraseExplanation(expIdx);
        assert(erased);
    }
    assert(part > 0);

    ::opensmt::vec<Formula> itps;
    auto interpolationContext = solver.getInterpolationContext();
    interpolationContext->getSingleInterpolant(itps, part);
    assert(itps.size() == 1);
    Formula itp = itps[0];

    auto const & logic = solver.getLogic();

    bool const itpIsConj = logic.isAnd(itp);
#ifndef NDEBUG
    bool const strongBoolItpAlg = config.boolInterpolationAlg == BoolInterpolationAlg::strong;
    auto const isLit = [&logic](auto & phi) {
        assert(not logic.isFalse(phi));
        assert(not logic.isTrue(phi));
        if (logic.isAtom(phi)) { return true; }
        if (not logic.isNot(phi)) { return false; }
        auto & phiTerm = logic.getPterm(phi);
        assert(phiTerm.size() == 1);
        return logic.isAtom(phiTerm[0]);
    };
    assert(not strongBoolItpAlg or itpIsConj or isLit(itp));
#endif

    if (not filteringVars and not itpIsConj) {
        assignNew<FormulaExplanation>(explanationPtr, fw, itp);
        return;
    }

    ConjunctExplanation newConjExplanation{fw};

    auto const insertItp = [&](Formula const & phi) {
        assert(logic.isOr(phi) or isLit(phi));
        auto phiexplanationPtr = std::make_unique<FormulaExplanation>(fw, phi);
        newConjExplanation.insertExplanation(std::move(phiexplanationPtr));
    };

    if (itpIsConj) {
        for (Formula const & phi : logic.getPterm(itp)) {
            insertItp(phi);
        }
    } else {
        insertItp(itp);
    }

    if (filteringVars) { newConjExplanation.merge(std::move(cexplanation)); }

    assignNew<ConjunctExplanation>(explanationPtr, std::move(newConjExplanation));
}
} // namespace xspace::expand::opensmt
