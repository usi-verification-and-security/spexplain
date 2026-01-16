#include "Network.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ranges>
#include <sstream>

namespace spexplain {
using namespace std::string_literals;

namespace {
    std::vector<std::string> split(std::string const & s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }
    template<typename TTransform>
    auto parseValues(std::string const & str, TTransform transformation, char delimiter = ',') {
        auto valueRange =
            split(str, delimiter) | std::ranges::views::transform([&](auto const & s) { return transformation(s); });
        using valType = std::invoke_result_t<TTransform, std::string>;
        std::vector<valType> values;
        std::ranges::copy(valueRange, std::back_inserter(values));
        return values;
    }

} // namespace

/// Load neural network from .nnet file.
/// \param filename the path to the .nnet file
/// \return In-memory representation of the network
/// The file begins with header lines, some information about the network architecture, normalization information, and then model parameters. Line by line:
/// 1: Header text. This can be any number of lines so long as they begin with "//"
/// 2: Four values: Number of layers, number of inputs, number of outputs, and maximum layer size
/// 3: A sequence of values describing the network layer sizes. Begin with the input size, then the size of the first layer, second layer, and so on until the output layer size
/// 4: A flag that is no longer used, can be ignored
/// 5: Minimum values of inputs (used to keep inputs within expected range)
/// 6: Maximum values of inputs (used to keep inputs within expected range)
/// 7: Mean values of inputs and one value for all outputs (used for normalization)
/// 8: Range values of inputs and one value for all outputs (used for normalization)
/// 9+: Begin defining the weight matrix for the first layer, followed by the bias vector. The weights and biases for the second layer follow after, until the weights and biases for the output layer are defined.
std::unique_ptr<Network> Network::fromNNetFile(std::string_view filename) {
    std::ifstream file{std::string{filename}};
    if (not file.good()) { throw std::ifstream::failure{"Could not open model file " + std::string{filename}}; }

    std::string line;

    // Skip header lines
    while (std::getline(file, line)) {
        if (line.find("//") == std::string::npos) { break; }
    }

    std::vector<std::string> networkArchitecture = split(line, ',');
    assert(networkArchitecture.size() == 4);
    std::size_t numLayers =
        std::stoull(networkArchitecture[0]) + 1; // The format does not include input layer, but we do
    std::size_t numInputs = std::stoull(networkArchitecture[1]);
    std::size_t numOutputs = std::stoull(networkArchitecture[2]);
    std::size_t maxLayerSize = std::stoull(networkArchitecture[3]);

    // Specify layer sizes
    std::getline(file, line);
    auto layer_sizes = parseValues(line, [](auto const & val) { return std::stoull(val); });
    assert(layer_sizes.size() == numLayers);

    // Flag that can be ignored
    std::getline(file, line);

    // Min and max input values
    auto transformation = [](auto const & val) { return std::stod(val); };
    std::getline(file, line);
    auto inputMinValues = static_cast<Values &&>(parseValues(line, transformation));
    std::getline(file, line);
    auto inputMaxValues = static_cast<Values &&>(parseValues(line, transformation));
    assert(inputMinValues.size() == numInputs and inputMaxValues.size() == numInputs);

    // Skip normalization information
    for (int i = 0; i < 2; i++) {
        std::getline(file, line);
    }

    // Parse model parameters
    Weights weights(numLayers);
    Biases biases(numLayers);
    for (auto layer = 0u; layer < numLayers - 1; layer++) {
        // Parse weights
        auto layerSize = layer_sizes.at(layer + 1);
        for (auto i = 0u; i < layerSize; i++) {
            std::getline(file, line);
            std::vector<std::string> weightStrings = split(line, ',');
            weights[layer].emplace_back();
            for (auto const & weightString : weightStrings) {
                weights[layer][i].push_back(std::stod(weightString));
            }
        }

        // Parse biases
        for (auto i = 0u; i < layerSize; i++) {
            std::getline(file, line);
            std::vector<std::string> biasStrings = split(line, ',');
            std::string biasString = biasStrings[0];
            biases[layer].push_back(std::stod(biasString));
        }
    }
    file.close();

    return std::unique_ptr<Network>{new Network(numInputs, numOutputs, numLayers, maxLayerSize,
                                                std::move(inputMinValues), std::move(inputMaxValues),
                                                std::move(weights), std::move(biases))};
}

std::size_t Network::nClasses() const {
    std::size_t nOutputs_ = nOutputs();
    assert(nOutputs_ > 0);
    assert(nOutputs_ != 2);
    if (nOutputs_ == 1) { return 2; }
    return nOutputs_;
}

std::size_t Network::getLayerSize(std::size_t layerNum) const {
    if (layerNum == 0) { return nInputs(); }
    assert(layerNum <= biases.size());
    return biases[layerNum - 1].size();
}

Network::Values const & Network::getWeights(std::size_t layerNum, std::size_t nodeIndex) const {
    assert(layerNum > 0);
    assert(nodeIndex < getLayerSize(layerNum));
    return weights[layerNum - 1][nodeIndex];
}

Float Network::getBias(std::size_t layerNum, std::size_t nodeIndex) const {
    assert(layerNum > 0);
    assert(nodeIndex < getLayerSize(layerNum));
    return biases[layerNum - 1][nodeIndex];
}

Float Network::getInputLowerBound(std::size_t nodeIndex) const {
    assert(nodeIndex < getLayerSize(0));
    return inputMinimums.at(nodeIndex);
}

Float Network::getInputUpperBound(std::size_t nodeIndex) const {
    assert(nodeIndex < getLayerSize(0));
    return inputMaximums.at(nodeIndex);
}

Network::Output Network::evaluate(Sample const & sample, EvalConfig const & conf) const {
    std::vector<Output::Values> allValues = computeOutputValues(sample, conf);
    assert(allValues.size() >= 1);
    assert(conf.storeHiddenNeuronValues or allValues.size() == 1);
    Output::Values values = std::move(allValues.back());
    assert(values.size() == nOutputs());

    Classification cls = computeClassification(values);

    std::vector<Output::Values> hiddenNeuronInputValues;
    std::vector<Output::Values> hiddenNeuronOutputValues;
    if (conf.storeHiddenNeuronValues) {
        allValues.pop_back();
        std::size_t const nHiddenLayers_ = nHiddenLayers();
        assert(allValues.size() == 2 * nHiddenLayers_);
        hiddenNeuronInputValues.reserve(nHiddenLayers_);
        hiddenNeuronOutputValues.reserve(nHiddenLayers_);
        for (std::size_t i = 0; i < nHiddenLayers_; ++i) {
            hiddenNeuronInputValues.push_back(std::move(allValues[i * 2]));
            hiddenNeuronOutputValues.push_back(std::move(allValues[i * 2 + 1]));
        }
    }

    return {.classification = std::move(cls),
            .values = std::move(values),
            .hiddenNeuronInputValues = std::move(hiddenNeuronInputValues),
            .hiddenNeuronOutputValues = std::move(hiddenNeuronOutputValues)};
}

std::vector<Network::Output::Values> Network::computeOutputValues(Sample const & sample,
                                                                  EvalConfig const & evalConf) const {
    std::size_t const nVars = nInputs();
    if (sample.size() != nVars) { throw std::logic_error("Input values do not have expected size!"); }

    bool const storeHiddenNeuronValues = evalConf.storeHiddenNeuronValues;

    static_assert(std::is_same_v<Values, Output::Values>);

    std::size_t const nLayers_ = nLayers();
    std::size_t const nHiddenLayers_ = nHiddenLayers();

    auto previousLayerValues = sample;
    Values currentLayerValues;
    std::vector<Values> outputValues;
    if (not storeHiddenNeuronValues) {
        outputValues.reserve(1);
    } else {
        outputValues.reserve(2 * nHiddenLayers_ + 1);
    }
    assert(nLayers_ == nHiddenLayers_ + 2);
    for (std::size_t layer = 1;; ++layer) {
        std::size_t const layerSize = getLayerSize(layer);
        for (std::size_t node = 0; node < layerSize; ++node) {
            auto const & incomingWeights = getWeights(layer, node);
            assert(incomingWeights.size() == previousLayerValues.size());
            Values addends;
            for (auto i = 0u; i < incomingWeights.size(); ++i) {
                addends.push_back(incomingWeights[i] * previousLayerValues[i]);
            }
            Float sum = std::accumulate(addends.begin(), addends.end(), getBias(layer, node));
            currentLayerValues.push_back(sum);
        }

        assert(layer <= nLayers_ - 1);
        if (layer == nLayers_ - 1) { break; }

        if (storeHiddenNeuronValues) { outputValues.push_back(currentLayerValues); }
        std::transform(currentLayerValues.begin(), currentLayerValues.end(), currentLayerValues.begin(),
                       [](Float val) { return std::max(Float{0}, val); });
        if (storeHiddenNeuronValues) { outputValues.push_back(currentLayerValues); }
        previousLayerValues = std::move(currentLayerValues);
        currentLayerValues.clear();
    }
    assert(not storeHiddenNeuronValues or outputValues.size() == 2 * nHiddenLayers_);

    outputValues.push_back(std::move(currentLayerValues));

    return outputValues;
}

Network::Classification Network::computeClassification(Output::Values const & values) const {
    assert(nClasses() >= 2);
    if (nClasses() == 2) {
        return computeBinaryClassification(values);
    } else {
        return computeNonBinaryClassification(values);
    }
}

Network::Classification Network::computeBinaryClassification(Output::Values const & values) const {
    assert(values.size() == 1);
    auto const val = values.front();
    Network::Classification::Label label = (val < 0) ? 0 : 1;

    assert(label == 0 or label == 1);
    assert(label == (val >= 0));

    return {.label = label};
}

Network::Classification Network::computeNonBinaryClassification(Output::Values const & values) const {
    auto const it = std::ranges::max_element(values);
    auto dist = std::distance(values.begin(), it);
    Classification::Label label = dist;
    return {.label = label};
}

void Network::Values::print(std::ostream & os) const {
    assert(not empty());
    os << front();
    for (Float val : *this | std::views::drop(1)) {
        os << ',' << val;
    }
}

Float getOutputValue(Network::Output::Values const & values, std::size_t nodeIndex) {
    assert(nodeIndex < values.size());
    return values[nodeIndex];
}

Float getHiddenNeuronValue(std::vector<Network::Output::Values> const & values, std::size_t layerNum,
                           std::size_t nodeIndex) {
    std::size_t const hiddenLayerIdx = layerNum - 1;
    if (hiddenLayerIdx >= values.size()) {
        throw std::out_of_range{"Hidden layer index is out of range: "s + std::to_string(layerNum) +
                                " >= " + std::to_string(values.size() + 1)};
    }

    auto & layerNeuronValues = values[hiddenLayerIdx];
    return getOutputValue(layerNeuronValues, nodeIndex);
}

bool activatedHiddenNeuron(Network::Output const & output, std::size_t layerNum, std::size_t nodeIndex) {
    Float const value = getHiddenNeuronOutputValue(output, layerNum, nodeIndex);
    assert(value >= 0);
    return value > Float{0};
}
} // namespace spexplain
