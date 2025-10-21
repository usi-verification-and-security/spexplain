
#ifndef XAI_SMT_NETWORKLAYER_H
#define XAI_SMT_NETWORKLAYER_H

// #include "Network.h"


#include <network/Network.h>
#include <vector>
#include <memory>
#include <spexplain/common/Core.h>

namespace spexplain {
class NetworkLayer {
public:
    struct Values : std::vector<Float> {
        using Idx = size_type;

        using vector::vector;

        void print(std::ostream &) const;
    };

    using Vector2D = std::vector<Values>;
    using Vector3D = std::vector<std::vector<Values>>;
    using Vector4D = std::vector<std::vector<std::vector<Values>>>;

    using WeightsVariant = std::variant<
        Values, // 1D
        Vector2D, // 2D
        Vector3D, // 3D
        Vector4D
        >;
    using Bias = Values;
    using LayerOutputVariant = std::variant<
        Values, // 1D
        Vector2D, // 2D
        Vector3D // 3D
        >;

    bool followedByRelu;

    virtual std::vector<std::size_t> getLayerSize() const = 0;
    virtual WeightsVariant getWeights() const = 0;
    virtual WeightsVariant getWeights(std::size_t) const = 0;
    virtual Bias getBias() const = 0;
    virtual LayerOutputVariant computeLayerOutput(LayerOutputVariant const & input) const = 0;
    virtual ~NetworkLayer() = default;

    bool followsByRelu() const { return followedByRelu; }
};
}

#endif // XAI_SMT_NETWORKLAYER_H


