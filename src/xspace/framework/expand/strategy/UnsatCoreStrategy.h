#ifndef XSPACE_EXPAND_UCORESTRATEGY_H
#define XSPACE_EXPAND_UCORESTRATEGY_H

#include "Strategy.h"

namespace xai::verifiers {
class UnsatCoreVerifier;
}

namespace xspace {
class ConjunctExplanation;
class IntervalExplanation;

class Framework::Expand::UnsatCoreStrategy : virtual public Strategy {
public:
    struct Config {
        bool splitIntervals = false;
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

    void executeBody(Explanations &, Network::Dataset const &, ExplanationIdx) override;
    virtual void executeBody(ConjunctExplanation &);
    virtual void executeBody(IntervalExplanation &);

    Config config{};
};
} // namespace xspace

#endif // XSPACE_EXPAND_UCORESTRATEGY_H
