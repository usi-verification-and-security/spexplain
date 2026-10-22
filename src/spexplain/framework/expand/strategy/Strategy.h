#ifndef SPEXPLAIN_EXPAND_STRATEGY_H
#define SPEXPLAIN_EXPAND_STRATEGY_H

#include "../Expand.h"

#include <spexplain/common/Bound.h>
#include <spexplain/common/Interval.h>

namespace spexplain {
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

    virtual void execute(Explanations &, Network::Dataset const &, Sample::Idx);

    bool checkFormsExplanation();

    void intersectExplanation(std::unique_ptr<Explanation> &, std::unique_ptr<Explanation> &&);

protected:
    struct AssertExplanationConf {
        bool ignoreVarOrder = false;
        bool splitIntervals = false;
    };

    xai::verifiers::Verifier const & getVerifier() const { return expand.getVerifier(); }
    xai::verifiers::Verifier & getVerifier() { return *expand.verifierPtr; }

    [[noreturn]] static void throwUnknownResultInternalException() { throw UnknownResultInternalException{}; }

    virtual bool storeNamedTerms() const { return false; }

    virtual void executeInit(Explanations &, Network::Dataset const &, Sample::Idx);
    virtual void executeBody(Explanations &, Network::Dataset const &, Sample::Idx) = 0;
    virtual void executeFinish(Explanations &, Network::Dataset const &, Sample::Idx);

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

    virtual bool intersectExplanationImpl(std::unique_ptr<Explanation> &, std::unique_ptr<Explanation> &);
    virtual bool intersectIntervalExplanationImpl(std::unique_ptr<Explanation> &, std::unique_ptr<Explanation> &);
    virtual bool intersectNonIntervalExplanationImpl(std::unique_ptr<Explanation> &, std::unique_ptr<Explanation> &);

    Expand & expand;

    VarOrdering varOrdering;

private:
    template<bool omitIdx = false>
    void assertIntervalExplanationTp(IntervalExplanation const &, AssertExplanationConf const &,
                                     VarIdx idxToOmit = invalidVarIdx);
};
} // namespace spexplain

#endif // SPEXPLAIN_EXPAND_STRATEGY_H
