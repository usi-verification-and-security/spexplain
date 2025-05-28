#ifndef XSPACE_EXPAND_NOPSTRATEGY_H
#define XSPACE_EXPAND_NOPSTRATEGY_H

#include "Strategy.h"

namespace xspace {
class Framework::Expand::NopStrategy : public Strategy {
public:
    using Strategy::Strategy;

    static char const * name() { return "nop"; }

protected:
    void executeBody(Explanations &, Network::Dataset const &, ExplanationIdx) {}
};
} // namespace xspace

#endif // XSPACE_EXPAND_NOPSTRATEGY_H
