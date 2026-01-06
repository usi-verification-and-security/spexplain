#include "Config.h"

#include <sstream>
#include <stdexcept>

using namespace std::string_literals;

namespace spexplain {
void Framework::Config::init(Network const & network) noexcept {
    networkPtr = &network;

    fixAllSampleNeuronActivationsMap.init(network);
    preferAllSampleNeuronActivationsMap.init(network);
}

void Framework::Config::fixSampleNeuronActivationAt(Sample::Idx idx, HiddenNeuronPosition const & pos) {
    assert(networkPtr);
    auto & map = fixSampleNeuronActivationMaps[idx];
    map.setNetwork(*networkPtr);
    map.insertOrAssign(pos.layer, pos.node, not pos.negated);
}

void Framework::Config::preferSampleNeuronActivationAt(Sample::Idx idx, HiddenNeuronPosition const & pos) {
    assert(networkPtr);
    auto & map = preferSampleNeuronActivationMaps[idx];
    map.setNetwork(*networkPtr);
    map.insertOrAssign(pos.layer, pos.node, not pos.negated);
}

[[nodiscard]]
std::optional<bool> Framework::Config::tryGetFixingOfSampleNeuronActivationAt(Sample::Idx idx, std::size_t layer,
                                                                              std::size_t node) const {
    if (auto mapIt = fixSampleNeuronActivationMaps.find(idx); mapIt != fixSampleNeuronActivationMaps.end()) {
        return mapIt->second.tryGetAt(layer, node);
    }

    return std::nullopt;
}

[[nodiscard]]
std::optional<bool> Framework::Config::tryGetPreferenceOfSampleNeuronActivationAt(Sample::Idx idx, std::size_t layer,
                                                                                  std::size_t node) const {
    if (auto mapIt = preferSampleNeuronActivationMaps.find(idx); mapIt != preferSampleNeuronActivationMaps.end()) {
        return mapIt->second.tryGetAt(layer, node);
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

namespace {
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
        if (not iss) { throw std::invalid_argument{"Unrecognized string of "s + msg + ": " + str}; }
        if (not iss.eof()) { throw std::invalid_argument{"Additional arguments after reading "s + msg + ": " + str}; }
        if constexpr (includeSampleV) {
            if (sampleIdx <= 0) { throw std::invalid_argument{"Sample index must be >= 1: "s + str}; }
        }
        if (layer <= 0) { throw std::invalid_argument{"Hidden layer must be >= 1: "s + str}; }
        if (node == 0) { throw std::invalid_argument{"Node must not be equal to 0: "s + str}; }

        Framework::Config::HiddenNeuronPosition pos{.layer = static_cast<std::size_t>(layer),
                                                    .node = static_cast<std::size_t>(std::abs(node) - 1),
                                                    .negated = node < 0};
        if constexpr (includeSampleV) {
            return {static_cast<std::size_t>(sampleIdx - 1), std::move(pos)};
        } else {
            return pos;
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
} // namespace spexplain
