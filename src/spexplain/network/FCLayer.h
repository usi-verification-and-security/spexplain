//
// Created by labbaf on 26.08.2025.
//

#ifndef SPEXPLAIN_FC_LAYER_H
#define SPEXPLAIN_FC_LAYER_H

#include "NetworkLayer.h"

#include <cstddef>
#include <vector>

namespace spexplain {
class FCLayer : public NetworkLayer {
public:
    FCLayer(Shape inputShape, Shape outputShape, Values weights, Values biases)
        : NetworkLayer{"fc", std::move(inputShape), std::move(outputShape)}, weights{std::move(weights)}, biases{std::move(biases)} {}

    Values computeLayerOutput(Values const & input) const override;

    Values const & getWeights() const { return weights; }
    Values const & getBiases() const { return biases; }

private:
    Values weights;
    Values biases;
};
} // namespace spexplain

#endif // SPEXPLAIN_FC_LAYER_H
