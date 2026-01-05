#include "Config.h"

#include <stdexcept>

using namespace std::string_literals;

namespace spexplain {
Framework::Config::DefaultSampleNeuronActivations makeDefaultSampleNeuronActivations(std::string_view str) {
    using enum Framework::Config::DefaultSampleNeuronActivations;
    if (str == "all") { return all; }
    if (str == "none") { return none; }
    if (str == "active") { return active; }
    if (str == "inactive") { return inactive; }

    throw std::invalid_argument{"Unrecognized sample neuron activations mode: "s + std::string{str}};
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
