#include "TransposeLayer.h"

#include <cassert>

namespace spexplain {

namespace {

std::vector<std::size_t> computeStrides(NetworkLayer::Shape const & shape)
{
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 0; --i)
    {
        if (i < shape.size())
            strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

} // namespace

TransposeLayer::TransposeLayer(Shape inputShape, Permutation perm)
    : NetworkLayer{"transpose", inputShape, permutedShape(inputShape, perm)}, perm{std::move(perm)}
{
}

TransposeLayer::Shape TransposeLayer::permutedShape(Shape const & inputShape, Permutation const & perm)
{
    Shape out;
    out.reserve(perm.size());
    for (std::size_t idx : perm)
        out.push_back(inputShape[idx]);
    return out;
}

std::vector<std::size_t> TransposeLayer::outputToInputIndexMap() const
{
    assert(inputShape.size() == perm.size());

    auto inStrides = computeStrides(inputShape);
    auto outStrides = computeStrides(outputShape);

    std::vector<std::size_t> map(getOutputSize(), 0);

    for (std::size_t outFlat = 0; outFlat < map.size(); ++outFlat)
    {
        std::size_t rem = outFlat;
        std::size_t inFlat = 0;

        for (std::size_t axis = 0; axis < outputShape.size(); ++axis)
        {
            std::size_t coord = rem / outStrides[axis];
            rem %= outStrides[axis];
            inFlat += coord * inStrides[perm[axis]];
        }

        map[outFlat] = inFlat;
    }

    return map;
}

NetworkLayer::Values TransposeLayer::computeLayerOutput(Values const & input) const
{
    assert(input.size() == getInputSize());
    assert(inputShape.size() == perm.size());

    auto const map = outputToInputIndexMap();

    Values output(getOutputSize(), Float{});
    for (std::size_t outFlat = 0; outFlat < output.size(); ++outFlat)
        output[outFlat] = input[map[outFlat]];

    return output;
}

} // namespace spexplain

