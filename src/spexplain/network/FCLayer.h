//
// Created by labbaf on 26.08.2025.
//

#ifndef XAI_SMT_FCLAYER_H
#define XAI_SMT_FCLAYER_H

#include <spexplain/network/NetworkLayer.h>
#include <vector>
#include <any>

namespace spexplain {
class FCLayer : public NetworkLayer {
public:
    using Weight = Vector2D;
    using Bias = Values;

    FCLayer(std::vector<std::size_t> size_, Weight weight_, Bias bias_, bool followedByRelu_ = true)
        : size{std::move(size_)},
          weights{std::move(weight_)},
          bias{std::move(bias_)} {
        this->followedByRelu = followedByRelu_;
    }

    std::vector<std::size_t> getLayerSize() const override { return size; }
    WeightsVariant getWeights() const override { return weights; }
    WeightsVariant getWeights(std::size_t nodeIndex) const override;
    Bias getBias() const override { return bias; }

    LayerOutputVariant computeLayerOutput(LayerOutputVariant const & input) const override;

private:
    std::vector<std::size_t> size;
    Weight weights;
    Bias bias;
};
}

#endif // XAI_SMT_FCLAYER_H
