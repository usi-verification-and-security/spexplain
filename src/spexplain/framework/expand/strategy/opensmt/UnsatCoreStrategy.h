#ifndef SPEXPLAIN_EXPAND_OSMTUCORESTRATEGY_H
#define SPEXPLAIN_EXPAND_OSMTUCORESTRATEGY_H

#include "../UnsatCoreStrategy.h"
#include "Strategy.h"

namespace spexplain::expand::opensmt {
class UnsatCoreStrategy : public Framework::Expand::UnsatCoreStrategy, public Strategy {
public:
    struct Config {
        //+ move to the base as well
        bool minimal = false;
    };

    using Framework::Expand::UnsatCoreStrategy::UnsatCoreStrategy;
    UnsatCoreStrategy(Framework::Expand & exp, Framework::Expand::UnsatCoreStrategy::Config const & baseConf,
                      Config const & conf, Framework::Expand::VarOrdering order = {})
        : Framework::Expand::Strategy{exp, std::move(order)},
          Framework::Expand::UnsatCoreStrategy{exp, baseConf},
          Strategy{exp}, // necessary but actually ignored by the compiler
          config{conf} {}

protected:
    using Strategy::getVerifier;

    void executeInit(Explanations &, Network::Dataset const &, ExplanationIdx) override;

    // using Strategy::assertFormulaExplanation allows to handle those explanations as well

    Config config{};
};
} // namespace spexplain::expand::opensmt

#endif // SPEXPLAIN_EXPAND_OSMTUCORESTRATEGY_H
