//
// Created by labbaf on 05.09.2025.
//

#ifndef XAI_SMT_FLATTENLAYER_H
#define XAI_SMT_FLATTENLAYER_H

#include <spexplain/network/NetworkLayer.h>


namespace spexplain {
class FlattenLayer : public NetworkLayer {
public:
    FlattenLayer() {
        this->followedByRelu = false;
    };

    std::vector<std::size_t> getLayerSize()  const override  { throw std::logic_error("Flatten layer does not have fixed sizes");}
    WeightsVariant getWeights() const override { throw std::logic_error("Flatten layer does not have weights");}
    WeightsVariant getWeights(std::size_t)  const override { throw std::logic_error("Flatten layer does not have weights");}
    Bias getBias()  const override { throw std::logic_error("Flatten layer does not have biases");}

    LayerOutputVariant computeLayerOutput(LayerOutputVariant const & input) const override;

};
}
#endif // XAI_SMT_FLATTENLAYER_H
