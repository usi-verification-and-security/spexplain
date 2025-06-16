#include "Framework.h"

#include "Config.h"
#include "Parse.h"
#include "Preprocess.h"
#include "Print.h"
#include "expand/Expand.h"

#include <spexplain/common/Macro.h>

// for the destructor
#include "expand/strategy/Strategy.h"

#include <verifiers/Verifier.h>

namespace spexplain {
Framework::Framework() : Framework(Config{}) {}

Framework::Framework(Config const & config) : configPtr{MAKE_UNIQUE(config)} {
    expandPtr = std::make_unique<Expand>(*this);
    preprocessPtr = std::make_unique<Preprocess>(*this);
    printPtr = std::make_unique<Print>(*this);
}

Framework::Framework(Config const & config, std::unique_ptr<Network> networkPtr_) : Framework(config) {
    setNetwork(std::move(networkPtr_));
}

Framework::Framework(Config const & config, std::unique_ptr<Network> networkPtr_, std::istream & expandStrategiesSpec)
    : Framework(config, std::move(networkPtr_)) {
    setExpand(expandStrategiesSpec);
}

Framework::Framework(Config const & config, std::unique_ptr<Network> networkPtr_, std::string_view verifierName,
                     std::istream & expandStrategiesSpec)
    : Framework(config, std::move(networkPtr_)) {
    setExpand(verifierName, expandStrategiesSpec);
}

Framework::~Framework() = default;

void Framework::setConfig(Config const & config) {
    configPtr = MAKE_UNIQUE(config);
}

void Framework::setNetwork(std::unique_ptr<Network> networkPtr_) {
    assert(networkPtr_);
    networkPtr = std::move(networkPtr_);

    auto & network = *networkPtr;
    std::size_t const size = network.getInputSize();
    varNames.reserve(size);
    domainIntervals.reserve(size);
    for (VarIdx idx = 0; idx < size; ++idx) {
        varNames.emplace_back(makeVarName(idx));
        domainIntervals.emplace_back(network.getInputLowerBound(idx), network.getInputUpperBound(idx));
    }
}

void Framework::setExpand(std::istream & strategiesSpec) {
    assert(expandPtr);
    expandPtr->setStrategies(strategiesSpec);
    expandPtr->setVerifier();
}

void Framework::setExpand(std::string_view verifierName, std::istream & strategiesSpec) {
    assert(expandPtr);
    expandPtr->setStrategies(strategiesSpec);
    expandPtr->setVerifier(verifierName);
}

void Framework::dumpClassificationsAsSmtLib2Queries() {
    //++ move outside of Expand -> probably also move the verifier right into the fw
    expandPtr->dumpClassificationsAsSmtLib2Queries();
}

Explanations Framework::explain(Network::Dataset & data) {
    auto & preprocess = getPreprocess();

    preprocess(data);
    auto explanations = preprocess.makeExplanationsFromSamples(data);

    expand(explanations, data);

    return explanations;
}

Explanations Framework::expand(std::string_view fileName, Network::Dataset & data) {
    auto & preprocess = getPreprocess();
    Parse parse{*this};

    preprocess(data);
    //++ only interval explanations supported, but general phis probably require a solver
    auto explanations = parse.parseIntervalExplanations(fileName, data);

    expand(explanations, data);

    return explanations;
}

void Framework::expand(Explanations & explanations, Network::Dataset const & data) {
    (*expandPtr)(explanations, data);
}
} // namespace spexplain
