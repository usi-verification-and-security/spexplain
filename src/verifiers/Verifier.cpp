#include "Verifier.h"

#include <spexplain/common/Macro.h>

#include <cassert>

namespace xai::verifiers {

namespace {
    void insertArgsToTp(auto & vectorOfMaps, LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                        auto &&... args) {
        assert(layer <= nHiddenLayers);
        assert(node < layerSize);
        vectorOfMaps.resize(nHiddenLayers + 1);
        auto & map = vectorOfMaps[layer];
        map.reserve(layerSize);
        map.emplace(node, FORWARD(args)...);
    }
} // namespace

void Verifier::fixNeuronActivation(LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                                   bool activation) {
    insertArgsToTp(fixedNeuronActivations, layer, node, nHiddenLayers, layerSize, activation);
}

void Verifier::preferNeuronActivation(LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                                      bool activation) {
    insertArgsToTp(preferredNeuronActivations, layer, node, nHiddenLayers, layerSize, activation);
}

namespace {
    template<typename T>
    std::optional<T> getArgFromMaps(std::vector<std::unordered_map<NodeIndex, T>> const & vectorOfMaps,
                                    LayerIndex layer, NodeIndex node) {
        if (vectorOfMaps.size() <= layer) { return std::nullopt; }

        auto const & map = vectorOfMaps[layer];
        if (auto it = map.find(node); it != map.end()) { return it->second; }

        return std::nullopt;
    }
} // namespace

std::optional<bool> Verifier::getFixedNeuronActivation(LayerIndex layer, NodeIndex node) const {
    return getArgFromMaps(fixedNeuronActivations, layer, node);
}

std::optional<bool> Verifier::getPreferredNeuronActivation(LayerIndex layer, NodeIndex node) const {
    return getArgFromMaps(preferredNeuronActivations, layer, node);
}

void Verifier::resetSample() {
    resetSampleQuery();
    checksCount = 0;
    fixedNeuronActivations.clear();
    preferredNeuronActivations.clear();
}

} // namespace xai::verifiers
