#include "Network.h"

#include "ConvLayer.h"
#include "FCLayer.h"
#include "FlattenLayer.h"
#include "NetworkLayer.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ranges>
#include <sstream>

namespace spexplain {
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

std::unique_ptr<Network> Network::buildDummyNetwork() {
    std::vector<std::unique_ptr<NetworkLayer>> layers;
    auto convLayer = std::make_unique<ConvLayer>(
        std::vector<std::size_t>{2, 2, 2}, // layer size (height, width, channels)
        ConvLayer::Weight{Vector3D(1,Vector2D{
            Values{1.0f, 1.0f}, Values{1.0f, 1.0f}}),
            Vector3D(1,Vector2D{
            Values{1.0f, -1.0f}, Values{1.0f, -1.0f}})}, // filter
        ConvLayer::Bias{0.01f, 0.0f}, // bias
        1,                     // stride
        0                      // padding
    );
    layers.push_back(std::move(convLayer));

    auto flattenLayer = std::make_unique<FlattenLayer>();
    layers.push_back(std::move(flattenLayer));

    auto fcLayer = std::make_unique<FCLayer>(
            std::vector<std::size_t>{1},
            FCLayer::Weight(1,
            Values{1.0f, -1.0f, 1.0f, -1.0f, 2.0f, 2.0f, 2.0f, 2.0f}), // weights for hidden FC // weights
            FCLayer::Bias{1.0f},
            false // not followed by ReLU
    );
    layers.push_back(std::move(fcLayer));

    // Define input and output sizes based on the layers
    std::vector<std::size_t> numInputs = {3,3}; // Example input size (3x3 image flattened)
    std::size_t numOutputs = 1;

    // Define input bounds
    InputVariant inputMinValues = InputVariant{Vector2D{{0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f}}};
    InputVariant inputMaxValues = InputVariant{Vector2D(3, Values(3, 10.0f))};


    // Create the Network instance
    return std::unique_ptr<Network>{new Network(numInputs, numOutputs, layers.size(), 9,
                                                    std::move(inputMinValues), std::move(inputMaxValues),
                                                    std::move(layers))}; // Make sure Network constructor accepts std::vector<std::unique_ptr<NetworkLayer>>
}


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
std::unique_ptr<Network> Network::fromNNet(std::string_view filename) {
     std::ifstream file{std::string{filename}};
     if (not file.good()) { throw std::ifstream::failure{"Could not open model file " + std::string{filename}}; }

     std::string line;

     // Skip header lines
     while (std::getline(file, line)) {
         if (line.find("//") == std::string::npos) { break; }
     }

     std::vector<std::string> networkArchitecture = split(line, ',');
     assert(networkArchitecture.size() == 4);
     std::size_t numLayers = std::stoull(networkArchitecture[0]) + 1; // The format does not include input layer, but we do
     std::vector<std::size_t> numInputs = {std::stoull(networkArchitecture[1])};
     std::size_t numOutputs = std::stoull(networkArchitecture[2]);
     std::size_t maxLayerSize = std::stoull(networkArchitecture[3]);

     // Specify layer sizes
     std::getline(file, line);
     auto layer_sizes = parseValues(line, [](auto const & val) { return std::stoull(val); });
     assert(layer_sizes.size() == numLayers);

     // Flag that can be ignored
     std::getline(file, line);

     // Min and max input values
     auto transformation = [](auto const & val) { return std::stof(val); };
     std::getline(file, line);
     auto inputMinValues = static_cast<Values &&>(parseValues(line, transformation));
     std::getline(file, line);
     auto inputMaxValues = static_cast<Values &&>(parseValues(line, transformation));
    // TODO: Faezeh: make the right assertion: it has the same shape
     // assert(inputMinValues.size() == numInputs and inputMaxValues.size() == numInputs);

     // Skip normalization information
     for (int i = 0; i < 2; i++) {
         std::getline(file, line);
     }

     // Parse model parameters
     std::vector<std::unique_ptr<NetworkLayer>> layers;
     for (auto layer = 0u; layer < numLayers - 1; layer++) {
         // Parse weights
         auto layerSize = layer_sizes.at(layer + 1);
         NetworkLayer::Vector2D layerWeights(layerSize);
         for (auto i = 0u; i < layerSize; i++) {
             std::getline(file, line);
             std::vector<std::string> weightStrings = split(line, ',');
             for (auto const & weightString : weightStrings) {
                 layerWeights[i].push_back(std::stof(weightString));
             }
         }
         // Parse biases
         NetworkLayer::Values layerBiases;
         for (auto i = 0u; i < layerSize; i++) {
             std::getline(file, line);
             std::vector<std::string> biasStrings = split(line, ',');
             std::string biasString = biasStrings[0];
             layerBiases.push_back(std::stof(biasString));
         }
         bool followedByRelu = true;
         if (layer == numLayers - 2) {
             followedByRelu = false;
         }
         auto fcLayer = std::make_unique<FCLayer>(
             std::vector<std::size_t>{layerSize},
             FCLayer::Weight{std::move(layerWeights)}, // weights for hidden FC // weights
             FCLayer::Bias{std::move(layerBiases)}, // biases for each output node
             followedByRelu
         );
         layers.push_back(std::move(fcLayer));
     }
     file.close();

    return std::unique_ptr<Network>{new Network(numInputs, numOutputs, layers.size(), 9,
                                                    std::move(inputMinValues), std::move(inputMaxValues),
                                                    std::move(layers))};
}


std::vector<size_t> Network::getLayerSize(std::size_t layerNum) const {
    if (layerNum == 0) return std::vector<size_t>{inputShape};;
    return getLayer(layerNum)->getLayerSize();
}

NetworkLayer const * Network::getLayer(std::size_t layerNum) const {
    assert(layerNum < numLayers);
    return layers.at(layerNum).get();
}

Float Network::getInputLowerBound(std::size_t nodeIndex) const {
    if (inputShape.size() == 1) {
        assert(nodeIndex < getLayerSize(0)[0]);
        assert(std::holds_alternative<Values>(inputMinimums));
        const auto & minValuse = std::get<Values>(inputMinimums);
        return minValuse.at(nodeIndex);
    } else if (inputShape.size() == 2) {
        std::size_t rows = inputShape[0];
        std::size_t cols = inputShape[1];
        assert(nodeIndex < rows * cols);
        std::size_t i = nodeIndex / cols;
        std::size_t j = nodeIndex % cols;
        return getInputLowerBound(i, j);
    } else if (inputShape.size() == 3) {
        std::size_t depth = inputShape[0];
        std::size_t rows = inputShape[1];
        std::size_t cols = inputShape[2];
        assert(nodeIndex < depth * rows * cols);
        std::size_t i = nodeIndex / (rows * cols);
        std::size_t j = (nodeIndex / cols) % rows;
        std::size_t k = nodeIndex % cols;
        return getInputLowerBound(i, j, k);
    } else {
        throw std::logic_error("Input dimension not supported for getInputLowerBound");
    }
}

Float Network::getInputUpperBound(std::size_t nodeIndex) const {
    if (inputShape.size() == 1) {
        assert(nodeIndex < getLayerSize(0)[0]);
        assert(std::holds_alternative<Values>(inputMaximums));
        const auto & minValuse = std::get<Values>(inputMaximums);
        return minValuse.at(nodeIndex);
    } else if (inputShape.size() == 2) {
        std::size_t rows = inputShape[0];
        std::size_t cols = inputShape[1];
        assert(nodeIndex < rows * cols);
        std::size_t i = nodeIndex / cols;
        std::size_t j = nodeIndex % cols;
        return getInputUpperBound(i, j);
    } else if (inputShape.size() == 3) {
        std::size_t depth = inputShape[0];
        std::size_t rows = inputShape[1];
        std::size_t cols = inputShape[2];
        assert(nodeIndex < depth * rows * cols);
        std::size_t i = nodeIndex / (rows * cols);
        std::size_t j = (nodeIndex / cols) % rows;
        std::size_t k = nodeIndex % cols;
        return getInputUpperBound(i, j, k);
    } else {
        throw std::logic_error("Input dimension not supported for getInputUpperBound");
    }
}

// 2D lower bound
Float Network::getInputLowerBound(std::size_t i, std::size_t j) const {
    assert(std::holds_alternative<Vector2D>(inputMinimums));
    const auto & mat = std::get<Vector2D>(inputMinimums);
    assert(i < mat.size() && j < mat[i].size());
    return mat[i][j];
}

Float Network::getInputUpperBound(std::size_t i, std::size_t j) const {
    assert(std::holds_alternative<Vector2D>(inputMaximums));
    const auto & mat = std::get<Vector2D>(inputMaximums);
    assert(i < mat.size() && j < mat[i].size());
    return mat[i][j];
}

// 3D lower bound
Float Network::getInputLowerBound(std::size_t i, std::size_t j, std::size_t k) const {
    assert(std::holds_alternative<Vector3D>(inputMinimums));
    const auto & tensor = std::get<Vector3D>(inputMinimums);
    assert(i < tensor.size() && j < tensor[i].size() && k < tensor[i][j].size());
    return tensor[i][j][k];
}

Float Network::getInputUpperBound(std::size_t i, std::size_t j, std::size_t k) const {
    assert(std::holds_alternative<Vector3D>(inputMaximums));
    const auto & tensor = std::get<Vector3D>(inputMaximums);
    assert(i < tensor.size() && j < tensor[i].size() && k < tensor[i][j].size());
    return tensor[i][j][k];
}


bool Network::isBinaryClassifier() const {
    assert(getOutputSize() != 0);
    assert(getOutputSize() != 2);
    return getOutputSize() == 1;
}

Network::Output Network::operator()(Sample const & sample) const {
    if (sample.size() != getInputSizeFlat()) {
        throw std::logic_error("Input values do not have expected size!");
    }
    auto sampleReshaped = reshapeInput(sample);
    Output::Values values = computeOutputValues(sampleReshaped);
    Classification cls = computeClassification(values);

    return {.classification = std::move(cls), .values = std::move(values)};
}

Network::Output::Values Network::computeOutputValues(InputVariant const & sample) const {
    auto inputSize = getInputSize();
    // if (sample.size() != inputSize) { throw std::logic_error("Input values do not have expected size!"); }

    auto currentLayerInput = sample;

    for (auto layerInd = 0u; layerInd < getNumLayers(); layerInd++) {
        InputVariant currentLayerOutput;
        auto layer = getLayer(layerInd);
        if (auto fcLayer = dynamic_cast<spexplain::FCLayer const *>(layer)) {
            currentLayerOutput = fcLayer->computeLayerOutput(currentLayerInput);
        } else if (auto convLayer = dynamic_cast<spexplain::ConvLayer const *>(layer)) {
            currentLayerOutput = convLayer->computeLayerOutput(currentLayerInput);
        } else if (auto flattenLayer = dynamic_cast<spexplain::FlattenLayer const *>(layer)) {
            currentLayerOutput = flattenLayer->computeLayerOutput(currentLayerInput);
        } else {
            throw std::logic_error("Unsupported layer type!");
        }
        currentLayerInput = std::move(currentLayerOutput);
    }
    assert(std::holds_alternative<Values>(currentLayerInput));
    auto const & currentLayerOutput = std::get<Values>(currentLayerInput);
    return currentLayerOutput;
}

Network::Classification Network::computeClassification(Output::Values const & values) const {
    if (isBinaryClassifier()) {
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

Network::InputVariant Network::reshapeInput(Network::InputVariant sample) const {
    assert(std::holds_alternative<Values>(sample));
    auto const & sample1D = std::get<Values>(sample);
    if (inputShape.size() == 1) {
        assert(sample1D.size() == inputShape[0]);
        return sample1D;
    } else if (inputShape.size() == 2) {
        assert(sample1D.size() == inputShape[0] * inputShape[1]);
        Network::Vector2D mat(inputShape[0], Network::Values(inputShape[1]));
        for (std::size_t i = 0; i < inputShape[0]; ++i) {
            for (std::size_t j = 0; j < inputShape[1]; ++j) {
                mat[i][j] = sample1D[i * inputShape[1] + j];
            }
        }
        return Network::InputVariant{mat};
    } else if (inputShape.size() == 3) {
        assert(sample1D.size() == inputShape[0] * inputShape[1] * inputShape[2]);
        Network::Vector3D tensor(inputShape[0], Network::Vector2D(inputShape[1], Network::Values(inputShape[2])));
        for (std::size_t i = 0; i < inputShape[0]; ++i) {
            for (std::size_t j = 0; j < inputShape[1]; ++j) {
                for (std::size_t k = 0; k < inputShape[2]; ++k) {
                    tensor[i][j][k] = sample1D[i * inputShape[1] * inputShape[2] + j * inputShape[2] + k];
                }
            }
        }
        return tensor;
    } else {
        throw std::logic_error("Input dimension not supported for reshapeInputVars");
    }
}
} // namespace spexplain
