#ifndef XSPACE_EXPAND_STRATEGY_H
#define XSPACE_EXPAND_STRATEGY_H

#include "../Expand.h"

#include <xspace/common/Bound.h>
#include <xspace/common/Interval.h>

namespace xspace {
class VarBound;
class PartialExplanation;
class ConjunctExplanation;
class IntervalExplanation;

class Framework::Expand::Strategy {
public:
    class Factory;

    Strategy(Expand &, VarOrdering = {});
    virtual ~Strategy() = default;

    static char const * name() = delete;

    virtual bool requiresSMTSolver() const { return false; }

    virtual void execute(Explanations &, Dataset const &, ExplanationIdx);

protected:
    struct AssertExplanationConf {
        bool ignoreVarOrder = false;
        bool splitIntervals = false;
    };

    xai::verifiers::Verifier const & getVerifier() const { return expand.getVerifier(); }
    xai::verifiers::Verifier & getVerifier() { return *expand.verifierPtr; }

    virtual bool storeNamedTerms() const { return false; }

    virtual void executeInit(Explanations &, Dataset const &, ExplanationIdx);
    virtual void executeBody(Explanations &, Dataset const &, ExplanationIdx) = 0;
    virtual void executeFinish(Explanations &, Dataset const &, ExplanationIdx);

    void initVarOrdering();

    void assertExplanation(PartialExplanation const &);
    void assertExplanation(PartialExplanation const &, AssertExplanationConf const &);
    virtual bool assertExplanationImpl(PartialExplanation const &, AssertExplanationConf const &);

    void assertConjunctExplanation(ConjunctExplanation const &);
    void assertConjunctExplanation(ConjunctExplanation const &, AssertExplanationConf const &);

    void assertIntervalExplanation(IntervalExplanation const &);
    void assertIntervalExplanation(IntervalExplanation const &, AssertExplanationConf const &);
    void assertIntervalExplanationExcept(IntervalExplanation const &, VarIdx);
    void assertIntervalExplanationExcept(IntervalExplanation const &, VarIdx, AssertExplanationConf const &);

    void assertVarBound(VarBound const &);
    void assertVarBound(VarBound const &, AssertExplanationConf const &);

    void assertInterval(VarIdx, Interval const &);
    void assertInnerInterval(VarIdx, Interval const &, bool splitIntervals = false);
    void assertInnerInterval(VarIdx, LowerBound const &, UpperBound const &, bool splitIntervals = false);
    void assertInnerIntervalNoSplit(VarIdx, LowerBound const &, UpperBound const &);
    void assertInnerIntervalSplit(VarIdx, LowerBound const &, UpperBound const &);
    void assertPoint(VarIdx, Float);
    void assertPoint(VarIdx, EqBound const &, bool splitIntervals = false);
    void assertPointNoSplit(VarIdx, EqBound const &);
    void assertPointSplit(VarIdx, EqBound const &);

    void assertBound(VarIdx, Bound const &);
    void assertLowerBound(VarIdx, LowerBound const &);
    void assertUpperBound(VarIdx, UpperBound const &);
    void assertEquality(VarIdx, EqBound const &);

    bool checkFormsExplanation();

    Expand & expand;

    VarOrdering varOrdering;

private:
    template<bool omitIdx = false>
    void assertIntervalExplanationTp(IntervalExplanation const &, AssertExplanationConf const &,
                                     VarIdx idxToOmit = invalidVarIdx);
};
} // namespace xspace

#endif // XSPACE_EXPAND_STRATEGY_H
