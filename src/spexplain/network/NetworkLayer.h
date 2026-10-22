#ifndef SPEXPLAIN_NETWORK_LAYER_H
#define SPEXPLAIN_NETWORK_LAYER_H

#include <spexplain/common/Core.h>

#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace spexplain {
class NetworkLayer {
public:
    using Values = std::vector<Float>;
    using Shape = std::vector<std::size_t>;

    explicit NetworkLayer(std::string type, Shape inputShape = {}, Shape outputShape = {})
        : type{std::move(type)}, inputShape{std::move(inputShape)}, outputShape{std::move(outputShape)} {}

    virtual ~NetworkLayer() = default;

    std::string const & getType() const { return type; }
    Shape const & getInputShape() const { return inputShape; }
    Shape const & getOutputShape() const { return outputShape; }

    std::size_t getInputSize() const { return tensorSize(inputShape); }
    std::size_t getOutputSize() const { return tensorSize(outputShape); }

    virtual Values computeLayerOutput(Values const & input) const = 0;

protected:
    static std::size_t tensorSize(Shape const & shape) {
        return shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>{});
    }

    std::string type;
    Shape inputShape;
    Shape outputShape;
};
} // namespace spexplain

#endif // SPEXPLAIN_NETWORK_LAYER_H


