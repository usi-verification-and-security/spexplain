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
    if (not guidedNeuronActivations.empty()) { return true; }

    return false;
}

bool Verifier::defaultEncodingOutputVars() const {
    if (not encodingNeuronVars()) { return false; }
    if (fixedNeuronActivations.empty()) { return false; }

    return true;
}

bool Verifier::defaultEncodingReluLowerBounds() const {
    if (not encodingNeuronVars()) { return false; }

    auto const & network = getNetwork();
    if (network.nHiddenLayers() > 1) { return false; }

    return true;
}

void Verifier::fixNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    fixedNeuronActivations.insertOrAssign(layer, node, activation);
}

void Verifier::guideNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    guidedNeuronActivations.insertOrAssign(layer, node, activation);
}

bool Verifier::tryFixNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    return fixedNeuronActivations.tryEmplace(layer, node, activation);
}

bool Verifier::tryGuideNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    return guidedNeuronActivations.tryEmplace(layer, node, activation);
}

std::optional<bool> Verifier::getFixedNeuronActivation(LayerIndex layer, NodeIndex node) const {
    return fixedNeuronActivations.tryGetAt(layer, node);
}

std::optional<bool> Verifier::getGuidedNeuronActivation(LayerIndex layer, NodeIndex node) const {
    return guidedNeuronActivations.tryGetAt(layer, node);
}

void Verifier::init(spexplain::Network const & nw) {
    initImpl(nw);
    reset();
}

void Verifier::initImpl(spexplain::Network const & nw) {
    networkPtr = &nw;

    fixedNeuronActivations.setNetwork(nw);
    guidedNeuronActivations.setNetwork(nw);
}

std::unique_ptr<spexplain::Explanation> Verifier::getSampleModelRestrictions(spexplain::Framework const &) {
    return {};
}

void Verifier::resetSample() {
    resetSampleQuery();
    checksCount = 0;
    fixedNeuronActivations.clear();
    guidedNeuronActivations.clear();
}

} // namespace xai::verifiers
