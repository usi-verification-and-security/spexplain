#include "Preprocess.h"

#include "explanation/IntervalExplanation.h"
#include "explanation/VarBound.h"

#include <spexplain/common/Macro.h>
#include <spexplain/network/Dataset.h>

#include <algorithm>
#include <cassert>
#include <concepts>

namespace spexplain {
namespace {
Network::Output::Values toNetworkValues(Network2::Values const & values) {
    return {values.begin(), values.end()};
}

std::vector<Network::Output::Values> toNetworkHiddenValues(std::vector<Network2::Values> const & valuesByLayer) {
    std::vector<Network::Output::Values> converted;
    converted.reserve(valuesByLayer.size());
    for (auto const & values : valuesByLayer) {
        converted.push_back(toNetworkValues(values));
    }
    return converted;
}

Network::Output toNetworkOutput(Network2::Output const & output) {
    return {.classification = {.label = output.classification.label},
            .values = toNetworkValues(output.values),
            .hiddenNeuronInputValues = toNetworkHiddenValues(output.hiddenNeuronInputValues),
            .hiddenNeuronOutputValues = toNetworkHiddenValues(output.hiddenNeuronOutputValues)};
}
} // namespace

Framework::Preprocess::Preprocess(Framework & fw) : framework{fw} {}

void Framework::Preprocess::operator()(Network::Dataset & dataset) const {
    assert(not framework.varNames.empty());

    auto const & samples = dataset.getSamples();
    std::size_t const size = dataset.size();
    assert(size == samples.size());
    assert(not samples.empty());
    Network::Dataset::Outputs outputs;
    outputs.reserve(size);

    if (framework.hasNetwork2()) {
        auto const & network2 = framework.getNetwork2();
        for (auto const & sample : samples) {
            assert(sample.size() == framework.varSize());
            Network2::Values const input{sample.begin(), sample.end()};
            outputs.push_back(toNetworkOutput(network2.evaluate(input, {.storeHiddenNeuronValues = true})));
        }
    } else {
        auto const & network = framework.getNetwork();
        for (auto const & sample : samples) {
            assert(sample.size() == framework.varSize());
            outputs.push_back(network.evaluate(sample, {.storeHiddenNeuronValues = true}));
        }
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

std::unique_ptr<Explanation> Framework::Preprocess::makeExplanationFromSample(Sample const & sample) const {
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
