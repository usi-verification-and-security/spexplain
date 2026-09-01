//
// Created by labbaf on 12.06.2026.
//

#ifndef SPEXPLAIN_NETWORK2_H
#define SPEXPLAIN_NETWORK2_H

#include "NetworkLayer.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace spexplain {
class Network2 {
public:
    using LayerPtr = std::unique_ptr<NetworkLayer>;
    using Layers = std::vector<LayerPtr>;
    using Values = NetworkLayer::Values;

    struct Classification {
        using Label = std::size_t;
        Label label;
    };

    struct Output {
        using Values = Network2::Values;

        Classification classification;
        Values values{};
        std::vector<Values> hiddenNeuronInputValues{};
        std::vector<Values> hiddenNeuronOutputValues{};
    };

    struct EvalConfig {
        bool storeHiddenNeuronValues{false};
    };

    static std::unique_ptr<Network2> fromONNXFile(std::string_view filename);

    Network2() = default;

    void setInputShape(NetworkLayer::Shape shape);
    void setOutputShape(NetworkLayer::Shape shape) { outputShape = std::move(shape); }
    void setInputMinimums(Values minimums);
    void setInputMaximums(Values maximums);

    /// A trailing sigmoid is strictly monotone, so it never changes the classification: it only maps
    /// the logit into (0,1). Since it is not encodable in linear arithmetic, it is dropped by default
    /// and the network output is the raw logit, exactly as for the NNet models.
    /// Disabling this keeps the sigmoid during evaluation, but then the network cannot be encoded.
    void setDropTrailingSigmoid(bool drop) { dropTrailingSigmoid = drop; }
    bool dropsTrailingSigmoid() const { return dropTrailingSigmoid; }

    /// Number of leading layers that both evaluation and encoding actually consider, i.e. all layers
    /// except the trailing sigmoids when `dropTrailingSigmoid` is set.
    std::size_t nEffectiveLayers() const;

    NetworkLayer::Shape const & getInputShape() const { return inputShape; }
    NetworkLayer::Shape const & getOutputShape() const { return outputShape; }
    Float getInputLowerBound(std::size_t idx) const;
    Float getInputUpperBound(std::size_t idx) const;

    Layers const & getLayers() const { return layers; }
    void addLayer(LayerPtr layer) { layers.push_back(std::move(layer)); }

    Output evaluate(Values const & input, EvalConfig const & conf) const;
    Output evaluate(Values const & input) const { return evaluate(input, {}); }

    std::size_t nInputs() const { return tensorSize(inputShape); }
    std::size_t nOutputs() const { return tensorSize(outputShape); }
    std::size_t nClasses() const;


private:
    static bool isActivationLayer(std::string const & type);
    std::size_t nEffectiveActivationLayers() const;
    std::vector<Values> computeOutputValues(Values const & input, EvalConfig const & conf) const;
    Classification computeClassification(Values const & values) const;

    static std::size_t tensorSize(NetworkLayer::Shape const & shape);

    NetworkLayer::Shape inputShape{};
    NetworkLayer::Shape outputShape{};
    Values inputMinimums{};
    Values inputMaximums{};
    Layers layers{};
    bool dropTrailingSigmoid{true};
};
} // namespace spexplain

#endif // SPEXPLAIN_NETWORK2_H
