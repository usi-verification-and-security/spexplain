#include "SliceStrategy.h"

#include <xspace/framework/Preprocess.h>
#include <xspace/framework/explanation/IntervalExplanation.h>

#include <cassert>

namespace xspace {
void Framework::Expand::SliceStrategy::executeBody(Explanations & explanations, Network::Dataset const & data,
                                                   ExplanationIdx idx) {
    auto & fw = expand.getFramework();
    auto & preprocess = fw.getPreprocess();

    auto & sample = data.getSample(idx);
    auto sampleExpPtr = preprocess.makeExplanationFromSample(sample);
    assert(dynamic_cast<IntervalExplanation *>(sampleExpPtr.get()));
    auto & sampleExp = static_cast<IntervalExplanation &>(*sampleExpPtr);

    auto & varIndices = config.varIndices;
    assert(varIndices.size() <= fw.varSize());

    for (VarIdx vidx : varIndices) {
        [[maybe_unused]] bool const success = sampleExp.eraseVarBound(vidx);
        assert(success);
    }

    auto & explanationPtr = getExplanationPtr(explanations, idx);
    intersectExplanation(explanationPtr, std::move(sampleExpPtr));
}
} // namespace xspace
