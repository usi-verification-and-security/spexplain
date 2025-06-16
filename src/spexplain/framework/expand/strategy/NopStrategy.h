#ifndef SPEXPLAIN_EXPAND_NOPSTRATEGY_H
#define SPEXPLAIN_EXPAND_NOPSTRATEGY_H

#include "Strategy.h"

namespace spexplain {
class Framework::Expand::NopStrategy : public Strategy {
public:
    using Strategy::Strategy;

    static char const * name() { return "nop"; }

protected:
    void executeBody(Explanations &, Network::Dataset const &, ExplanationIdx) {}
};
} // namespace spexplain

#endif // SPEXPLAIN_EXPAND_NOPSTRATEGY_H
