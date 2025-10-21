
#ifndef XAI_SMT_CONVLAYER_H
#define XAI_SMT_CONVLAYER_H

#include <memory>
#include <string_view>
#include <spexplain/network/NetworkLayer.h>
#include <vector>
#include <any>

namespace spexplain {
class ConvLayer : public NetworkLayer {
public:
    using Weight = Vector4D;
    using Bias = Values;

    ConvLayer(std::vector<std::size_t> size_, Weight filter_, Bias bias_, int stride_, int padding_ = 0, bool followedByRelu_ = true)
        : size{size_},
          filters{std::move(filter_)},
          bias{std::move(bias_)},
          stride{stride_},
          padding{padding_} {
        this->followedByRelu = followedByRelu_;
    }

    std::vector<std::size_t> getLayerSize() const override { return size; }
    WeightsVariant getWeights() const override { return filters; }
    WeightsVariant getWeights(std::size_t kernelIndex) const override;
    Bias getBias() const override { return bias; }

    LayerOutputVariant computeLayerOutput(LayerOutputVariant const & input) const override;

    int getStride() const { return stride; }
    int getPadding() const { return padding; }

private:
    std::vector<std::size_t> size;
    Weight filters;
    Bias bias;
    int stride;
    int padding;

    Vector3D padInput(const ConvLayer::Vector3D& input) const;
};
}

#endif // XAI_SMT_CONVLAYER_H
