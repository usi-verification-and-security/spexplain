#include "Preprocess.h"

#include "explanation/IntervalExplanation.h"
#include "explanation/VarBound.h"

#include <spexplain/common/Macro.h>
#include <spexplain/network/Dataset.h>

#include <algorithm>
#include <cassert>
#include <concepts>

namespace spexplain {
Framework::Preprocess::Preprocess(Framework & fw) : framework{fw} {}

void Framework::Preprocess::operator()(Network::Dataset & dataset) const {
    assert(not framework.varNames.empty());

    auto const & network = framework.getNetwork();

    auto const & samples = dataset.getSamples();
    std::size_t const size = dataset.size();
    assert(size == samples.size());
    assert(not samples.empty());
    Network::Dataset::Outputs outputs;
    outputs.reserve(size);
    for (auto const & sample : samples) {
        assert(sample.size() == framework.varSize());
        Network::Output output = network(sample);
        outputs.push_back(std::move(output));
    }

    assert(outputs.size() == size);
    dataset.setComputedOutputs(std::move(outputs));
}

Explanations Framework::Preprocess::makeExplanationsFromSamples(Network::Dataset const & dataset) const {
    auto const & samples = dataset.getSamples();
    std::size_t const size = dataset.size();
    assert(size == samples.size());
    Explanations explanations;
    explanations.reserve(size);
    for (auto const & sample : samples) {
        auto explanationPtr = makeExplanationFromSample(sample);
        explanations.push_back(std::move(explanationPtr));
    }

    assert(explanations.size() == size);
    return explanations;
}

std::unique_ptr<Explanation> Framework::Preprocess::makeExplanationFromSample(Network::Sample const & sample) const {
    std::size_t const vSize = framework.varSize();

    assert(sample.size() == vSize);

    IntervalExplanation iexplanation{framework};
    for (VarIdx idx = 0; idx < vSize; ++idx) {
        Float val = sample[idx];
        iexplanation.insertVarBound(VarBound{framework, idx, val});
    }

    return MAKE_UNIQUE(std::move(iexplanation));
}
} // namespace spexplain
