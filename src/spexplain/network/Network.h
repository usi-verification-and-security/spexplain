#ifndef SPEXPLAIN_NETWORK_H
#define SPEXPLAIN_NETWORK_H

#include "NetworkLayer.h"

#include <spexplain/common/Core.h>

#include <memory>
#include <string_view>
#include <vector>
#include <numeric>

namespace spexplain {
class Network {
public:
    using Values = NetworkLayer::Values;
    using Vector3D = NetworkLayer::Vector3D;
    using Vector2D = NetworkLayer::Vector2D;


    struct Classification {
        using Label = std::size_t;

        Label label;
    };

    using Classifications = std::vector<Classification>;

    using Sample = Values;

    using InputVariant = std::variant<
        Values, // 1D
        Vector2D, // 2D
        Vector3D // 3D
    >;

    struct Output {
        using Values = Network::Values;

        Classification classification;
        Values values{};
    };

    class Dataset;

    static std::unique_ptr<Network> fromNNetFile(std::string_view filename);
    static std::unique_ptr<Network> fromNNet(std::string_view filename);
    // static std::unique_ptr<Network> fromONNXFile(std::string_view filename);
    static std::unique_ptr<Network> buildDummyNetwork();

    std::vector<size_t> getInputSize() const { return inputShape; }
    std::size_t getInputSizeFlat() const {
        return std::accumulate(inputShape.begin(), inputShape.end(), 1u, std::multiplies<>());
    }
    std::size_t getOutputSize() const { return numOutputs; }
    std::size_t getNumLayers() const { return numLayers; }
    std::vector<size_t> getLayerSize(std::size_t layerNum) const;

    NetworkLayer const * getLayer(std::size_t layerNum) const;

    Float getInputLowerBound(std::size_t node) const;
    Float getInputUpperBound(std::size_t node) const;
    Float getInputLowerBound(std::size_t i, std::size_t j) const;
    Float getInputUpperBound(std::size_t i, std::size_t j) const;
    Float getInputLowerBound(std::size_t i, std::size_t j, std::size_t k) const;
    Float getInputUpperBound(std::size_t i, std::size_t j, std::size_t k) const;


    bool isBinaryClassifier() const;

    Output operator()(Sample const &) const;

protected:
    Output::Values computeOutputValues(InputVariant const &) const;

    Classification computeClassification(Output::Values const &) const;
    Classification computeBinaryClassification(Output::Values const &) const;
    Classification computeNonBinaryClassification(Output::Values const &) const;

private:

    Network(std::vector<size_t> numInputs_, std::size_t numOutputs_, std::size_t numLayers_, std::size_t maxLayerSize_,
            InputVariant inputMinimums_, InputVariant inputMaximums_, std::vector<std::unique_ptr <NetworkLayer>> layers_)
        : inputShape{numInputs_},
          numOutputs{numOutputs_},
          numLayers{numLayers_},
          maxLayerSize{maxLayerSize_},
          inputMinimums{std::move(inputMinimums_)},
          inputMaximums{std::move(inputMaximums_)},
          layers(std::move(layers_))  {}

    std::vector<size_t> inputShape;
    std::size_t numOutputs;
    std::size_t numLayers;
    std::size_t maxLayerSize;
    InputVariant inputMinimums;
    InputVariant inputMaximums;
    std::vector<std::unique_ptr<NetworkLayer>> layers;

    InputVariant reshapeInput(InputVariant) const;

};
} // namespace spexplain

#endif // SPEXPLAIN_NETWORK_H
