#ifndef SPEXPLAIN_EXPAND_ABDUCTIVESTRATEGY_H
#define SPEXPLAIN_EXPAND_ABDUCTIVESTRATEGY_H

#include "Strategy.h"

namespace spexplain {
class Framework::Expand::AbductiveStrategy : public Strategy {
public:
    using Strategy::Strategy;

    static char const * name() { return "abductive"; }

protected:
    void executeBody(Explanations &, Network::Dataset const &, ExplanationIdx) override;
};
} // namespace spexplain

#endif // SPEXPLAIN_EXPAND_ABDUCTIVESTRATEGY_H
