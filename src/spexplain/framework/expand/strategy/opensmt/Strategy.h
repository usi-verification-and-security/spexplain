#ifndef SPEXPLAIN_EXPAND_OSMTSTRATEGY_H
#define SPEXPLAIN_EXPAND_OSMTSTRATEGY_H

#include "../Strategy.h"

namespace xai::verifiers {
class OpenSMTVerifier;
}

namespace spexplain::opensmt {
class FormulaExplanation;
}

namespace spexplain::expand::opensmt {
using namespace spexplain::opensmt;

//+ make templated with Formula
class Strategy : virtual public Framework::Expand::Strategy {
public:
    using Base = Framework::Expand::Strategy;

    using Base::Base;

    bool requiresSMTSolver() const override { return true; }

protected:
    xai::verifiers::OpenSMTVerifier const & getVerifier() const;
    xai::verifiers::OpenSMTVerifier & getVerifier();

    bool assertExplanationImpl(PartialExplanation const &, AssertExplanationConf const &) override;

    void assertFormulaExplanation(FormulaExplanation const &);
    void assertFormulaExplanation(FormulaExplanation const &, AssertExplanationConf const &);

    bool intersectNonIntervalExplanationImpl(std::unique_ptr<Explanation> &, std::unique_ptr<Explanation> &) override;
};
} // namespace spexplain::expand::opensmt

#endif // SPEXPLAIN_EXPAND_OSMTSTRATEGY_H
