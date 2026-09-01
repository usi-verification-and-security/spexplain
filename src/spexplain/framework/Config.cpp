#include "Config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std::string_literals;

namespace spexplain {
void Framework::Config::init(Network const & network) noexcept {
    networkPtr = &network;
    network2Ptr = nullptr;
    fixAllSampleNeuronActivationsMap.clear();
    preferAllSampleNeuronActivationsMap.clear();
    fixSampleNeuronActivationMaps.clear();
    preferSampleNeuronActivationMaps.clear();
}

void Framework::Config::init2(Network2 const & network2) noexcept {
    networkPtr = nullptr;
    network2Ptr = &network2;
    fixAllSampleNeuronActivationsMap.clear();
    preferAllSampleNeuronActivationsMap.clear();
    fixSampleNeuronActivationMaps.clear();
    preferSampleNeuronActivationMaps.clear();
}

void Framework::Config::setActivation(ActivationMap & activations, HiddenNeuronPosition const & pos) {
    activations[pos.layer].insert_or_assign(pos.node, not pos.negated);
}

std::optional<bool> Framework::Config::tryGetActivation(ActivationMap const & activations, std::size_t layer,
                                                        std::size_t node) {
    if (auto layerIt = activations.find(layer); layerIt != activations.end()) {
        if (auto nodeIt = layerIt->second.find(node); nodeIt != layerIt->second.end()) { return nodeIt->second; }
    }
    return std::nullopt;
}

std::size_t Framework::Config::nHiddenLayers() const {
    if (network2Ptr) {
        std::size_t reluLayers = 0;
        for (auto const & layerPtr : getNetwork2().getLayers()) {
            if (layerPtr and layerPtr->getType() == "relu") { ++reluLayers; }
        }
        return reluLayers;
    }
    return getNetwork().nHiddenLayers();
}

std::size_t Framework::Config::hiddenLayerSize(std::size_t layer) const {
    using namespace std::string_literals;

    if (layer == 0) { throw std::out_of_range{"Hidden layer is out of range: 0"}; }

    if (network2Ptr) {
        std::size_t reluLayer = 0;
        for (auto const & layerPtr : getNetwork2().getLayers()) {
            if (not layerPtr || layerPtr->getType() != "relu") { continue; }
            ++reluLayer;
            if (reluLayer == layer) { return layerPtr->getOutputSize(); }
        }
        throw std::out_of_range{"Hidden layer is out of range: "s + std::to_string(layer) + " > " +
                                std::to_string(reluLayer)};
    }

    return getNetwork().getLayerSize(layer);
}

void Framework::Config::validateHiddenNeuronPosition(HiddenNeuronPosition const & pos) const {
    using namespace std::string_literals;

    std::size_t const nHidden = nHiddenLayers();
    if (pos.layer == 0 || pos.layer > nHidden) {
        throw std::out_of_range{"Hidden layer is out of range: "s + std::to_string(pos.layer) + " > " +
                                std::to_string(nHidden)};
    }

    std::size_t const layerSize = hiddenLayerSize(pos.layer);
    if (pos.node >= layerSize) {
        throw std::out_of_range{"Node is out of range: "s + std::to_string(pos.node) + " >= " +
                                std::to_string(layerSize)};
    }
}

void Framework::Config::fixAllSampleNeuronActivationsAt(HiddenNeuronPosition const & pos) {
    validateHiddenNeuronPosition(pos);
    setActivation(fixAllSampleNeuronActivationsMap, pos);
}

void Framework::Config::preferAllSampleNeuronActivationsAt(HiddenNeuronPosition const & pos) {
    validateHiddenNeuronPosition(pos);
    setActivation(preferAllSampleNeuronActivationsMap, pos);
}

void Framework::Config::fixSampleNeuronActivationAt(Sample::Idx idx, HiddenNeuronPosition const & pos) {
    validateHiddenNeuronPosition(pos);
    setActivation(fixSampleNeuronActivationMaps[idx], pos);
}

void Framework::Config::preferSampleNeuronActivationAt(Sample::Idx idx, HiddenNeuronPosition const & pos) {
    validateHiddenNeuronPosition(pos);
    setActivation(preferSampleNeuronActivationMaps[idx], pos);
}

std::optional<bool> Framework::Config::tryGetFixingOfAllSampleNeuronActivationsAt(std::size_t layer,
                                                                                  std::size_t node) const {
    return tryGetActivation(fixAllSampleNeuronActivationsMap, layer, node);
}

std::optional<bool> Framework::Config::tryGetPreferenceOfAllSampleNeuronActivationsAt(std::size_t layer,
                                                                                      std::size_t node) const {
    return tryGetActivation(preferAllSampleNeuronActivationsMap, layer, node);
}

[[nodiscard]]
std::optional<bool> Framework::Config::tryGetFixingOfSampleNeuronActivationAt(Sample::Idx idx, std::size_t layer,
                                                                              std::size_t node) const {
    if (auto mapIt = fixSampleNeuronActivationMaps.find(idx); mapIt != fixSampleNeuronActivationMaps.end()) {
        return tryGetActivation(mapIt->second, layer, node);
    }

    return std::nullopt;
}

[[nodiscard]]
std::optional<bool> Framework::Config::tryGetPreferenceOfSampleNeuronActivationAt(Sample::Idx idx, std::size_t layer,
                                                                                  std::size_t node) const {
    if (auto mapIt = preferSampleNeuronActivationMaps.find(idx); mapIt != preferSampleNeuronActivationMaps.end()) {
        return tryGetActivation(mapIt->second, layer, node);
    }

    return std::nullopt;
}

Framework::Config::DefaultSampleNeuronActivations makeDefaultSampleNeuronActivations(std::string_view str) {
    using enum Framework::Config::DefaultSampleNeuronActivations;
    if (str == "all") { return all; }
    if (str == "none") { return none; }
    if (str == "active") { return active; }
    if (str == "inactive") { return inactive; }

    throw std::invalid_argument{"Unrecognized sample neuron activations mode: "s + std::string{str}};
}

std::string
defaultSampleNeuronActivationsToString(Framework::Config::DefaultSampleNeuronActivations sampleNeuronActivations) {
    using enum Framework::Config::DefaultSampleNeuronActivations;
    switch (sampleNeuronActivations) {
        case all:
            return "all";
        default:
        case none:
            return "none";
        case active:
            return "active";
        case inactive:
            return "inactive";
    }
}

namespace {
    std::size_t parsedSampleToIdx(long sampleIdx) {
        if (sampleIdx <= 0) {
            throw std::invalid_argument{"Sample index "s + std::to_string(sampleIdx) + " must be >= 1"};
        }

        return static_cast<std::size_t>(sampleIdx - 1);
    }

    std::size_t parsedHiddenLayerToIdx(long layer) {
        if (layer <= 0) { throw std::invalid_argument{"Hidden layer "s + std::to_string(layer) + " must be >= 1"}; }

        return static_cast<std::size_t>(layer);
    }

    Framework::Config::HiddenNeuronPosition parsedNodeToIdx(long node, std::size_t layerIdx) {
        if (node == 0) { throw std::invalid_argument{"Node must not be equal to 0"}; }

        return {.layer = layerIdx, .node = static_cast<std::size_t>(std::abs(node) - 1), .negated = node < 0};
    }

    Framework::Config::HiddenNeuronPosition parsedHiddenLayerAndNodeToIdx(long layer, long node) {
        return parsedNodeToIdx(node, parsedHiddenLayerToIdx(layer));
    }

    template<bool includeSampleV>
    std::conditional_t<includeSampleV, std::pair<Sample::Idx, Framework::Config::HiddenNeuronPosition>,
                       Framework::Config::HiddenNeuronPosition>
    makeHiddenNeuronPositionImpl(std::string_view sw) {
        std::string str{sw};
        std::istringstream iss{str};
        [[maybe_unused]] long sampleIdx;
        long layer;
        long node;
        char c;
        if constexpr (includeSampleV) { iss >> sampleIdx >> c; }
        iss >> layer >> c >> node;

        static constexpr char const * msg = [] {
            if constexpr (includeSampleV) {
                return "sample index, hidden layer and node";
            } else {
                return "hidden layer and node";
            }
        }();
        if (iss.fail()) { throw std::invalid_argument{"Unrecognized string of "s + msg + ": " + str}; }
        if (not iss.eof()) { throw std::invalid_argument{"Additional arguments after reading "s + msg + ": " + str}; }

        if constexpr (includeSampleV) {
            return {parsedSampleToIdx(sampleIdx), parsedHiddenLayerAndNodeToIdx(layer, node)};
        } else {
            return parsedHiddenLayerAndNodeToIdx(layer, node);
        }
    }
} // namespace

Framework::Config::HiddenNeuronPosition makeHiddenNeuronPosition(std::string_view sw) {
    return makeHiddenNeuronPositionImpl<false>(sw);
}

std::pair<Sample::Idx, Framework::Config::HiddenNeuronPosition> makeHiddenNeuronPositionOfSample(std::string_view sw) {
    return makeHiddenNeuronPositionImpl<true>(sw);
}

bool usingSampleNeuronActivations(Framework::Config::DefaultSampleNeuronActivations sampleNeuronActivations,
                                  bool sampleActivated) {
    using enum Framework::Config::DefaultSampleNeuronActivations;
    switch (sampleNeuronActivations) {
        case all:
            return true;
        default:
        case none:
            return false;
        case active:
            return sampleActivated;
        case inactive:
            return not sampleActivated;
    }
}

namespace {
    template<bool fixV>
    void parseSampleNeuronActivationsImpl(Framework::Config & config, std::string_view fileName) {
        std::ifstream ifs{std::string{fileName}};
        if (not ifs.good()) {
            throw std::ifstream::failure{"Could not open sample neuron activations file: "s + std::string{fileName}};
        }

        std::size_t const nHiddenLayers = config.nHiddenLayers();

        long parsedSampleIdx;
        long lastParsedSampleIdx = -1;
        long parsedNode;
        while (ifs >> parsedSampleIdx) {
            std::size_t const sampleIdx = parsedSampleToIdx(parsedSampleIdx);
            lastParsedSampleIdx = parsedSampleIdx;

            for (std::size_t layerIdx = 1; layerIdx < nHiddenLayers + 1; ++layerIdx) {
                for (;;) {
                    ifs >> parsedNode;
                    if (not ifs.good()) {
                        throw std::logic_error{"At sample "s + std::to_string(parsedSampleIdx) + ", layer " +
                                               std::to_string(layerIdx) + ": Expected number"};
                    }
                    if (parsedNode == 0) { break; }

                    Framework::Config::HiddenNeuronPosition pos = parsedNodeToIdx(parsedNode, layerIdx);
                    if constexpr (fixV) {
                        config.fixSampleNeuronActivationAt(sampleIdx, pos);
                    } else {
                        config.preferSampleNeuronActivationAt(sampleIdx, pos);
                    }
                }
            }
        }

        if (ifs.eof()) { return; }

        throw std::logic_error{
            (lastParsedSampleIdx == -1 ? ""s : "After sample "s + std::to_string(lastParsedSampleIdx) + ": ") +
            "Expected sample index"};
    }
} // namespace

void parseFixSampleNeuronActivations(Framework::Config & config, std::string_view fileName) {
    parseSampleNeuronActivationsImpl<true>(config, fileName);
}

void parsePreferSampleNeuronActivations(Framework::Config & config, std::string_view fileName) {
    parseSampleNeuronActivationsImpl<false>(config, fileName);
}
} // namespace spexplain
