#ifndef SPEXPLAIN_EXPAND_SLICESTRATEGY_H
#define SPEXPLAIN_EXPAND_SLICESTRATEGY_H

#include "Strategy.h"

namespace spexplain {
class Framework::Expand::SliceStrategy : public Strategy {
public:
    struct Config {
        std::vector<VarIdx> varIndices{};
    };

    using Strategy::Strategy;
    SliceStrategy(Framework::Expand & exp, Config const & conf, Framework::Expand::VarOrdering order = {})
        : Strategy{exp, std::move(order)},
          config{conf} {}

    static char const * name() { return "slice"; }

protected:
    void executeBody(Explanations &, Network::Dataset const &, ExplanationIdx) override;

    Config config{};
};
} // namespace spexplain

#endif // SPEXPLAIN_EXPAND_SLICESTRATEGY_H
