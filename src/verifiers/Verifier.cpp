#include "Verifier.h"

#include <spexplain/common/Macro.h>
#include <spexplain/framework/explanation/Explanation.h>

#include <cassert>

namespace xai::verifiers {

namespace {
    template<bool overwriteV>
    bool insertArgsToTp(auto & vectorOfMaps, LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                        auto &&... args) {
        using VectorOfMaps = std::remove_cvref_t<decltype(vectorOfMaps)>;
        using Map = typename VectorOfMaps::value_type;

        assert(layer <= nHiddenLayers);
        assert(node < layerSize);
        vectorOfMaps.resize(nHiddenLayers + 1);
        auto & map = vectorOfMaps[layer];
        map.reserve(layerSize);
        if constexpr (overwriteV) {
            map.insert_or_assign(node, typename Map::mapped_type{FORWARD(args)...});
            return true;
        } else {
            auto [it, inserted] = map.try_emplace(node, FORWARD(args)...);
            return inserted;
        }
    }
} // namespace

void Verifier::fixNeuronActivation(LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                                   bool activation) {
    insertArgsToTp<true>(fixedNeuronActivations, layer, node, nHiddenLayers, layerSize, activation);
}

void Verifier::preferNeuronActivation(LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                                      bool activation) {
    insertArgsToTp<true>(preferredNeuronActivations, layer, node, nHiddenLayers, layerSize, activation);
}

bool Verifier::tryFixNeuronActivation(LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                                      bool activation) {
    return insertArgsToTp<false>(fixedNeuronActivations, layer, node, nHiddenLayers, layerSize, activation);
}

bool Verifier::tryPreferNeuronActivation(LayerIndex layer, NodeIndex node, size_t nHiddenLayers, size_t layerSize,
                                         bool activation) {
    return insertArgsToTp<false>(preferredNeuronActivations, layer, node, nHiddenLayers, layerSize, activation);
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
