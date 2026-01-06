#ifndef SPEXPLAIN_EXPAND_UCORESTRATEGY_H
#define SPEXPLAIN_EXPAND_UCORESTRATEGY_H

#include "Strategy.h"

namespace xai::verifiers {
class UnsatCoreVerifier;
}

namespace spexplain {
class ConjunctExplanation;
class IntervalExplanation;

class Framework::Expand::UnsatCoreStrategy : virtual public Strategy {
public:
    struct Config {
        bool splitIntervals = false;
        std::vector<VarIdx> varIndicesFilter{};
    };

    using Strategy::Strategy;
    UnsatCoreStrategy(Expand & exp, Config const & conf, VarOrdering order = {})
        : Strategy{exp, std::move(order)},
          config{conf} {}

    static char const * name() { return "ucore"; }

    // Does not strictly require SMT solver
    using Strategy::requiresSMTSolver;

protected:
    xai::verifiers::UnsatCoreVerifier const & getVerifier() const;
    xai::verifiers::UnsatCoreVerifier & getVerifier();

    bool storeNamedTerms() const override { return true; }

    void executeInit(Explanations &, Network::Dataset const &, Sample::Idx) override;
    void executeBody(Explanations &, Network::Dataset const &, Sample::Idx) override;
    virtual void executeBody(ConjunctExplanation &);
    virtual void executeBody(IntervalExplanation &);

    Config config{};
};
} // namespace spexplain

#endif // SPEXPLAIN_EXPAND_UCORESTRATEGY_H
