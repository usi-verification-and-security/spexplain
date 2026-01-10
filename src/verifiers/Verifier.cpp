#include "Verifier.h"

#include <spexplain/framework/explanation/Explanation.h>

#include <cassert>

namespace xai::verifiers {

spexplain::Network const & Verifier::getNetwork() const {
    assert(networkPtr);
    return *networkPtr;
}

bool Verifier::defaultEncodingNeuronVars() const {
    if (not fixedNeuronActivations.empty()) { return true; }
    if (not preferredNeuronActivations.empty()) { return true; }

    return false;
}

bool Verifier::defaultEncodingOutputVars() const {
    if (not encodingNeuronVars()) { return false; }
    if (fixedNeuronActivations.empty()) { return false; }

    return true;
}

void Verifier::fixNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    fixedNeuronActivations.insertOrAssign(layer, node, activation);
}

void Verifier::preferNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    preferredNeuronActivations.insertOrAssign(layer, node, activation);
}

bool Verifier::tryFixNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    return fixedNeuronActivations.tryEmplace(layer, node, activation);
}

bool Verifier::tryPreferNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    return preferredNeuronActivations.tryEmplace(layer, node, activation);
}

std::optional<bool> Verifier::getFixedNeuronActivation(LayerIndex layer, NodeIndex node) const {
    return fixedNeuronActivations.tryGetAt(layer, node);
}

std::optional<bool> Verifier::getPreferredNeuronActivation(LayerIndex layer, NodeIndex node) const {
    return preferredNeuronActivations.tryGetAt(layer, node);
}

void Verifier::init(spexplain::Network const & nw) {
    initImpl(nw);
    reset();
}

void Verifier::initImpl(spexplain::Network const & nw) {
    networkPtr = &nw;

    fixedNeuronActivations.setNetwork(nw);
    preferredNeuronActivations.setNetwork(nw);
}

std::unique_ptr<spexplain::Explanation> Verifier::getSampleModelRestrictions(spexplain::Framework const &) {
    return {};
}

void Verifier::resetSample() {
    resetSampleQuery();
    checksCount = 0;
    fixedNeuronActivations.clear();
    preferredNeuronActivations.clear();
}

} // namespace xai::verifiers
