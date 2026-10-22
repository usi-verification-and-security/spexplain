#ifndef SPEXPLAIN_FLATTEN_LAYER_H
#define SPEXPLAIN_FLATTEN_LAYER_H

#include "NetworkLayer.h"

namespace spexplain {

class FlattenLayer : public NetworkLayer {
public:
    FlattenLayer(Shape inputShape, Shape outputShape)
        : NetworkLayer{"flatten", std::move(inputShape), std::move(outputShape)} {}

    Values computeLayerOutput(Values const & input) const override { return input; }
};

} // namespace spexplain

#endif // SPEXPLAIN_FLATTEN_LAYER_H

