//
// Created by labbaf on 12.06.2026.
//

#include "Network2.h"

#include "OnnxParser.h"

#include <cassert>
#include <fstream>
#include <ranges>
#include <stdexcept>

namespace spexplain {
std::size_t Network2::tensorSize(NetworkLayer::Shape const & shape) {
    std::size_t size = 1;
    for (auto dim : shape) { size *= dim; }
    return shape.empty() ? 0 : size;
}

void Network2::setInputShape(NetworkLayer::Shape shape) {
    inputShape = std::move(shape);
    std::size_t const inputSize = nInputs();

    if (inputMinimums.empty()) {
        inputMinimums.assign(inputSize, 0);
    } else if (inputMinimums.size() != inputSize) {
        throw std::logic_error("Input minimum tensor size does not match input shape");
    }

    if (inputMaximums.empty()) {
        inputMaximums.assign(inputSize, 1);
    } else if (inputMaximums.size() != inputSize) {
        throw std::logic_error("Input maximum tensor size does not match input shape");
    }
}

void Network2::setInputMinimums(Values minimums) {
    if (!inputShape.empty() && minimums.size() != nInputs()) {
        throw std::logic_error("Input minimum tensor size does not match input shape");
    }
    inputMinimums = std::move(minimums);
}

void Network2::setInputMaximums(Values maximums) {
    if (!inputShape.empty() && maximums.size() != nInputs()) {
        throw std::logic_error("Input maximum tensor size does not match input shape");
    }
    inputMaximums = std::move(maximums);
}

Float Network2::getInputLowerBound(std::size_t idx) const {
    assert(idx < nInputs());
    if (inputMinimums.empty()) { return 0; }
    return inputMinimums.at(idx);
}

Float Network2::getInputUpperBound(std::size_t idx) const {
    assert(idx < nInputs());
    if (inputMaximums.empty()) { return 1; }
    return inputMaximums.at(idx);
}

std::size_t Network2::nClasses() const {
    std::size_t const nOutputs_ = nOutputs();
    assert(nOutputs_ > 0);
    assert(nOutputs_ != 2);
    if (nOutputs_ == 1) { return 2; }
    return nOutputs_;
}

bool Network2::isActivationLayer(std::string const & type) {
    return type == "relu" || type == "sigmoid";
}

std::size_t Network2::nEffectiveLayers() const {
    std::size_t end = layers.size();
    if (not dropTrailingSigmoid) { return end; }
    while (end > 0 and layers[end - 1] and layers[end - 1]->getType() == "sigmoid") { --end; }
    return end;
}

std::size_t Network2::nEffectiveActivationLayers() const {
    std::size_t const end = nEffectiveLayers();
    std::size_t count = 0;
    for (std::size_t li = 0; li < end; ++li) {
        if (layers[li] && isActivationLayer(layers[li]->getType())) { ++count; }
    }
    return count;
}

std::vector<Network2::Values> Network2::computeOutputValues(Values const & input, EvalConfig const & conf) const {
    if (tensorSize(inputShape) != 0 && input.size() != tensorSize(inputShape)) {
        throw std::logic_error("Input values do not have expected size!");
    }

    Values current = input;
    std::vector<Values> allValues;

    // Trailing sigmoids are excluded (by default), so that the produced output is the raw logit and
    // both the classification and the stored activations match what OpenSMTVerifier2 encodes.
    std::size_t const end = nEffectiveLayers();

    if (!conf.storeHiddenNeuronValues) {
        allValues.reserve(1);
    } else {
        allValues.reserve(2 * nEffectiveActivationLayers() + 1);
    }

    //TODO:Faezeh You might want to store all the layer outputs, not just pre/post activations!
    // You might want to store them in a map instead of a vector
    for (std::size_t li = 0; li < end; ++li) {
        auto const & layer = layers[li];
        assert(layer);
        if (conf.storeHiddenNeuronValues && isActivationLayer(layer->getType())) {
            allValues.push_back(current);
        }
        current = layer->computeLayerOutput(current);
        if (conf.storeHiddenNeuronValues && isActivationLayer(layer->getType())) {
            allValues.push_back(current);
        }
    }

    allValues.push_back(std::move(current));
    return allValues;
}

Network2::Classification Network2::computeClassification(Values const & values) const {
    if (values.empty()) {
        throw std::logic_error("Cannot classify empty output vector");
    }

    // Same semantics as Network::computeClassification: for a single output, the value is a raw logit
    // (any trailing sigmoid has been dropped by computeOutputValues), so 0 is the decision threshold.
    if (values.size() == 1) {
        Classification::Label label = (values.front() < 0) ? 0 : 1;
        return {.label = label};
    }

    auto const it = std::ranges::max_element(values);
    return {.label = static_cast<Classification::Label>(std::distance(values.begin(), it))};
}

Network2::Output Network2::evaluate(Values const & input, EvalConfig const & conf) const {
    std::vector<Output::Values> allValues = computeOutputValues(input, conf);
    assert(!allValues.empty());
    assert(conf.storeHiddenNeuronValues || allValues.size() == 1);

    Output::Values values = std::move(allValues.back());
    if (tensorSize(outputShape) != 0) {
        assert(values.size() == tensorSize(outputShape));
    }

    Classification cls = computeClassification(values);

    std::vector<Output::Values> hiddenNeuronInputValues;
    std::vector<Output::Values> hiddenNeuronOutputValues;
    if (conf.storeHiddenNeuronValues) {
        allValues.pop_back();

        std::size_t const nActivationLayers = nEffectiveActivationLayers();
        assert(allValues.size() == 2 * nActivationLayers);

        hiddenNeuronInputValues.reserve(nActivationLayers);
        hiddenNeuronOutputValues.reserve(nActivationLayers);
        for (std::size_t i = 0; i < nActivationLayers; ++i) {
            hiddenNeuronInputValues.push_back(std::move(allValues[i * 2]));
            hiddenNeuronOutputValues.push_back(std::move(allValues[i * 2 + 1]));
        }
    }

    return {.classification = std::move(cls),
            .values = std::move(values),
            .hiddenNeuronInputValues = std::move(hiddenNeuronInputValues),
            .hiddenNeuronOutputValues = std::move(hiddenNeuronOutputValues)};
}

/// Load neural network from .onnx file.
/// \param filename the path to the .onnx file
/// \return In-memory representation of the network
std::unique_ptr<Network2> Network2::fromONNXFile(std::string_view filename) {
    return OnnxParser::buildNetwork2(filename);
}
} // namespace spexplain
