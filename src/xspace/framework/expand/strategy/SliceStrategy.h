#ifndef XSPACE_EXPAND_SLICESTRATEGY_H
#define XSPACE_EXPAND_SLICESTRATEGY_H

#include "Strategy.h"

namespace xspace {
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
} // namespace xspace

#endif // XSPACE_EXPAND_SLICESTRATEGY_H
