//
// Created by labbaf on 27.08.2025.
//

#include "FCLayer.h"


namespace spexplain {
FCLayer::LayerOutputVariant FCLayer::computeLayerOutput(LayerOutputVariant const & input) const {
    assert(std::holds_alternative<Values>(input));
    Values const & inputValues = std::get<Values>(input);
    assert(inputValues.size() == weights[0].size());
    assert(size[0] == weights.size());
    Values outputValues(size[0], 0.0f);
//TODO: Size usage here is wronge. we have only one dimension.
    for (std::size_t j = 0; j < size[0]; ++j) {
        Float sum = bias[j];
        for (std::size_t i = 0; i < inputValues.size(); ++i) {
            sum += weights[j][i] * inputValues[i];
        }
        float val = sum;
        if (followedByRelu && val < 0.0f) {
            val = 0.0f;
        }
        outputValues[j] = val;
    }
    return outputValues;
}

FCLayer::WeightsVariant FCLayer::getWeights(std::size_t nodeIndex) const {
    assert(nodeIndex < weights.size());
    return weights[nodeIndex];
}
}