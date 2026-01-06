#ifndef SPEXPLAIN_EXPAND_OSMTITPSTRATEGY_H
#define SPEXPLAIN_EXPAND_OSMTITPSTRATEGY_H

#include "Strategy.h"

#include <vector>

namespace spexplain::expand::opensmt {
class InterpolationStrategy : public Strategy {
public:
    enum class BoolInterpolationAlg { weak, strong };
    enum class ArithInterpolationAlg { weak, weaker, strong, stronger, factor };

    struct Config {
        BoolInterpolationAlg boolInterpolationAlg{BoolInterpolationAlg::strong};
        ArithInterpolationAlg arithInterpolationAlg{ArithInterpolationAlg::weak};
        float arithInterpolationAlgFactor{};
        std::vector<VarIdx> varIndicesFilter{};
    };

    using Strategy::Strategy;
    InterpolationStrategy(Framework::Expand & exp, Config const & conf, Framework::Expand::VarOrdering order = {})
        : Strategy::Base{exp, std::move(order)},
          config{conf} {}

    static char const * name() { return "itp"; }

protected:
    void executeInit(Explanations &, Network::Dataset const &, Sample::Idx) override;
    void executeBody(Explanations &, Network::Dataset const &, Sample::Idx) override;

    Config config{};
};
} // namespace spexplain::expand::opensmt

#endif // SPEXPLAIN_EXPAND_OSMTITPSTRATEGY_H
