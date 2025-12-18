#ifndef SPEXPLAIN_NETWORK_H
#define SPEXPLAIN_NETWORK_H

#include <spexplain/common/Core.h>

#include <memory>
#include <string_view>
#include <vector>

namespace spexplain {
class Network {
public:
    struct Values : std::vector<Float> {
        using Idx = size_type;

        using vector::vector;

        void print(std::ostream &) const;
    };

    struct Classification {
        using Label = std::size_t;

        Label label;
    };

    using Classifications = std::vector<Classification>;

    using Sample = Values;

    struct Output {
        using Values = Network::Values;

        Classification classification;
        Values values{};

        std::vector<Values> hiddenNeuronValues{};
    };

    class Dataset;

    static std::unique_ptr<Network> fromNNetFile(std::string_view filename);

    std::size_t nInputs() const { return numInputs; }
    std::size_t nOutputs() const { return numOutputs; }
    std::size_t nLayers() const { return numLayers; }
    std::size_t nHiddenLayers() const { return nLayers() - 2; }

    std::size_t nClasses() const;

    std::size_t getLayerSize(std::size_t layerNum) const;

    Values const & getWeights(std::size_t layerNum, std::size_t nodeIndex) const;

    Float getBias(std::size_t layerNum, std::size_t nodeIndex) const;

    Float getInputLowerBound(std::size_t node) const;
    Float getInputUpperBound(std::size_t node) const;

    struct EvalConfig {
        bool storeHiddenNeuronValues{false};
    };
    Output operator()(Sample const & sample) const { return evaluate(sample); }
    Output evaluate(Sample const & sample, EvalConfig const &) const;
    Output evaluate(Sample const & sample) const { return evaluate(sample, {}); }

protected:
    std::vector<Output::Values> computeOutputValues(Sample const &, EvalConfig const &) const;

    Classification computeClassification(Output::Values const &) const;
    Classification computeBinaryClassification(Output::Values const &) const;
    Classification computeNonBinaryClassification(Output::Values const &) const;

private:
    using Weights = std::vector<std::vector<Values>>;
    using Biases = std::vector<Values>;

    Network(std::size_t numInputs_, std::size_t numOutputs_, std::size_t numLayers_, std::size_t maxLayerSize_,
            Values inputMinimums_, Values inputMaximums_, Weights weights_, Biases biases_)
        : numInputs{numInputs_},
          numOutputs{numOutputs_},
          numLayers{numLayers_},
          maxLayerSize{maxLayerSize_},
          inputMinimums{std::move(inputMinimums_)},
          inputMaximums{std::move(inputMaximums_)},
          weights{std::move(weights_)},
          biases{std::move(biases_)} {}

    std::size_t numInputs;
    std::size_t numOutputs;
    std::size_t numLayers;
    std::size_t maxLayerSize;
    Values inputMinimums;
    Values inputMaximums;
    Weights weights;
    Biases biases;
};

Float getOutputValue(Network::Output::Values const &, std::size_t nodeIndex);
inline Float getOutputValue(Network::Output const & output, std::size_t nodeIndex) {
    return getOutputValue(output.values, nodeIndex);
}

Float getHiddenNeuronValue(Network::Output const &, std::size_t layerNum, std::size_t nodeIndex);
bool activatedHiddenNeuron(Network::Output const &, std::size_t layerNum, std::size_t nodeIndex);
} // namespace spexplain

#endif // SPEXPLAIN_NETWORK_H
