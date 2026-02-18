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

Framework::Framework(Config const & config, std::unique_ptr<Network> networkPtr_, std::istream & expandStrategiesSpec)
    : Framework(config) {
    setNetwork(std::move(networkPtr_));
    setStrategies(expandStrategiesSpec);
    setVerifier();
}

Framework::~Framework() = default;

void Framework::setConfig(Config const & config) {
    configPtr = MAKE_UNIQUE(config);
}

void Framework::setNetwork(std::unique_ptr<Network> networkPtr_) {
    assert(networkPtr_);
    networkPtr = std::move(networkPtr_);

    auto & network = *networkPtr;
    std::size_t const nVars = network.nInputs();
    varNames.reserve(nVars);
    domainIntervals.reserve(nVars);
    for (VarIdx idx = 0; idx < nVars; ++idx) {
        varNames.emplace_back(makeVarName(idx));
        domainIntervals.emplace_back(network.getInputLowerBound(idx), network.getInputUpperBound(idx));
    }
}

void Framework::setStrategies() {
    assert(expandPtr);
    expandPtr->setStrategies();
}

void Framework::setStrategies(std::istream & strategiesSpec) {
    assert(expandPtr);
    expandPtr->setStrategies(strategiesSpec);
}

void Framework::setVerifier() {
    assert(expandPtr);
    expandPtr->setVerifier();
}

void Framework::setVerifier(std::string_view name) {
    assert(expandPtr);
    expandPtr->setVerifier(name);
}

void Framework::setExplanationsFileName(std::string_view fileName) {
    assert(printPtr);
    printPtr->setExplanationsFileName(fileName);
}

void Framework::setStatsFileName(std::string_view fileName) {
    assert(printPtr);
    printPtr->setStatsFileName(fileName);
}

void Framework::setTimesFileName(std::string_view fileName) {
    assert(printPtr);
    printPtr->setTimesFileName(fileName);
}

void Framework::dumpDomainsAsSmtLib2Query() {
    //++ move outside of Expand -> probably also move the verifier right into the fw
    expandPtr->dumpDomainsAsSmtLib2Query();
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
