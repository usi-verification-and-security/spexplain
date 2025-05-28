#ifndef XSPACE_EXPAND_OSMTITPSTRATEGY_H
#define XSPACE_EXPAND_OSMTITPSTRATEGY_H

#include "Strategy.h"

#include <vector>

namespace xspace::expand::opensmt {
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
    void executeInit(Explanations &, Network::Dataset const &, ExplanationIdx) override;
    void executeBody(Explanations &, Network::Dataset const &, ExplanationIdx) override;

    Config config{};
};
} // namespace xspace::expand::opensmt

#endif // XSPACE_EXPAND_OSMTITPSTRATEGY_H
