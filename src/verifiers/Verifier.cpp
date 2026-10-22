#include "Verifier.h"

#include <spexplain/framework/explanation/Explanation.h>
#include <spexplain/network/Network2.h>

#include <cassert>
#include <stdexcept>

namespace {
void validateNetwork2NeuronPosition(spexplain::Network2 const & network2, xai::verifiers::LayerIndex layer,
                                    xai::verifiers::NodeIndex node) {
    using namespace std::string_literals;

    xai::verifiers::LayerIndex reluLayerIndex = 1;
    for (auto const & layerPtr : network2.getLayers()) {
        if (!layerPtr || layerPtr->getType() != "relu") { continue; }
        if (reluLayerIndex == layer) {
            if (node >= layerPtr->getOutputSize()) {
                throw std::out_of_range{"Node is out of range: "s + std::to_string(node) + " >= " +
                                        std::to_string(layerPtr->getOutputSize())};
            }
            return;
        }
        ++reluLayerIndex;
    }

    throw std::out_of_range{"Hidden layer is out of range: "s + std::to_string(layer) + " > " +
                            std::to_string(reluLayerIndex - 1)};
}
} // namespace

namespace xai::verifiers {

spexplain::Network const & Verifier::getNetwork() const {
    assert(networkPtr);
    return *networkPtr;
}

spexplain::Network2 const & Verifier::getNetwork2() const {
    assert(network2Ptr);
    return *network2Ptr;
}

bool Verifier::defaultEncodingNeuronVars() const {
    if (hasFixedNeuronActivations()) { return true; }
    if (hasPreferredNeuronActivations()) { return true; }

    return false;
}

bool Verifier::defaultEncodingOutputVars() const {
    if (not encodingNeuronVars()) { return false; }
    if (not hasFixedNeuronActivations()) { return false; }

    return true;
}

bool Verifier::defaultEncodingReluLowerBounds() const {
    if (not encodingNeuronVars()) { return false; }

    auto const & network = getNetwork();
    if (network.nHiddenLayers() > 1) { return false; }

    return true;
}

void Verifier::fixNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    if (network2Ptr) {
        validateNetwork2NeuronPosition(getNetwork2(), layer, node);
        fixedNeuronActivations2.insert_or_assign({layer, node}, activation);
        return;
    }
    fixedNeuronActivations.insertOrAssign(layer, node, activation);
}

void Verifier::preferNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    if (network2Ptr) {
        validateNetwork2NeuronPosition(getNetwork2(), layer, node);
        preferredNeuronActivations2.insert_or_assign({layer, node}, activation);
        return;
    }
    preferredNeuronActivations.insertOrAssign(layer, node, activation);
}

bool Verifier::tryFixNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    if (network2Ptr) {
        validateNetwork2NeuronPosition(getNetwork2(), layer, node);
        auto [it, inserted] = fixedNeuronActivations2.try_emplace({layer, node}, activation);
        (void)it;
        return inserted;
    }
    return fixedNeuronActivations.tryEmplace(layer, node, activation);
}

bool Verifier::tryPreferNeuronActivation(LayerIndex layer, NodeIndex node, bool activation) {
    if (network2Ptr) {
        validateNetwork2NeuronPosition(getNetwork2(), layer, node);
        auto [it, inserted] = preferredNeuronActivations2.try_emplace({layer, node}, activation);
        (void)it;
        return inserted;
    }
    return preferredNeuronActivations.tryEmplace(layer, node, activation);
}

std::optional<bool> Verifier::getFixedNeuronActivation(LayerIndex layer, NodeIndex node) const {
    if (network2Ptr) {
        if (auto it = fixedNeuronActivations2.find({layer, node}); it != fixedNeuronActivations2.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    return fixedNeuronActivations.tryGetAt(layer, node);
}

std::optional<bool> Verifier::getPreferredNeuronActivation(LayerIndex layer, NodeIndex node) const {
    if (network2Ptr) {
        if (auto it = preferredNeuronActivations2.find({layer, node}); it != preferredNeuronActivations2.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    return preferredNeuronActivations.tryGetAt(layer, node);
}

bool Verifier::hasFixedNeuronActivations() const {
    if (network2Ptr) { return not fixedNeuronActivations2.empty(); }
    return not fixedNeuronActivations.empty();
}

bool Verifier::hasPreferredNeuronActivations() const {
    if (network2Ptr) { return not preferredNeuronActivations2.empty(); }
    return not preferredNeuronActivations.empty();
}

void Verifier::init(spexplain::Network const & nw) {
    initImpl(nw);
    reset();
}

void Verifier::init(spexplain::Network2 const & nw) {
    initImpl(nw);
    reset();
}

void Verifier::initImpl(spexplain::Network const & nw) {
    networkPtr = &nw;
    network2Ptr = nullptr;

    fixedNeuronActivations.setNetwork(nw);
    preferredNeuronActivations.setNetwork(nw);
}

void Verifier::initImpl(spexplain::Network2 const & nw) {
    networkPtr = nullptr;
    network2Ptr = &nw;
}

std::unique_ptr<spexplain::Explanation> Verifier::getSampleModelRestrictions(spexplain::Framework const &) {
    return {};
}

void Verifier::resetSample() {
    resetSampleQuery();
    checksCount = 0;
    fixedNeuronActivations.clear();
    preferredNeuronActivations.clear();
    fixedNeuronActivations2.clear();
    preferredNeuronActivations2.clear();
}

} // namespace xai::verifiers
