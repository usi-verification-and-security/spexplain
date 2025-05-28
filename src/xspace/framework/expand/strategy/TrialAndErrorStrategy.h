#ifndef XSPACE_EXPAND_TRIALSTRATEGY_H
#define XSPACE_EXPAND_TRIALSTRATEGY_H

#include "Strategy.h"

namespace xspace {
class Framework::Expand::TrialAndErrorStrategy : public Strategy {
public:
    struct Config {
        int maxAttempts = 4;
    };

    using Strategy::Strategy;
    TrialAndErrorStrategy(Expand & exp, Config const & conf, VarOrdering order = {})
        : Strategy{exp, std::move(order)},
          config{conf} {}

    static char const * name() { return "trial"; }

protected:
    void executeBody(Explanations &, Network::Dataset const &, ExplanationIdx) override;

    Config config{};
};
} // namespace xspace

#endif // XSPACE_EXPAND_TRIALSTRATEGY_H
