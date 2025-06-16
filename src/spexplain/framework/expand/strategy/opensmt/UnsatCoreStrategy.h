#ifndef SPEXPLAIN_EXPAND_OSMTUCORESTRATEGY_H
#define SPEXPLAIN_EXPAND_OSMTUCORESTRATEGY_H

#include "../UnsatCoreStrategy.h"
#include "Strategy.h"

namespace spexplain::expand::opensmt {
class UnsatCoreStrategy : public Framework::Expand::UnsatCoreStrategy, public Strategy {
public:
    using Base = Framework::Expand::UnsatCoreStrategy;

    struct Config {
        bool minimal = false;
    };

    using Base::Base;
    UnsatCoreStrategy(Framework::Expand & exp, Base::Config const & baseConf, Config const & conf,
                      Framework::Expand::VarOrdering order = {})
        : Strategy::Base{exp, std::move(order)},
          Base{exp, baseConf},
          config{conf} {}

protected:
    using Strategy::getVerifier;

    void executeInit(Explanations &, Network::Dataset const &, ExplanationIdx) override;

    // using Strategy::assertFormulaExplanation allows to handle those explanations as well

    Config config{};
};
} // namespace spexplain::expand::opensmt

#endif // SPEXPLAIN_EXPAND_OSMTUCORESTRATEGY_H
