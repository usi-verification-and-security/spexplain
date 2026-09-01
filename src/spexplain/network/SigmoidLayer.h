#ifndef SPEXPLAIN_SIGMOID_LAYER_H
#define SPEXPLAIN_SIGMOID_LAYER_H

#include "NetworkLayer.h"

namespace spexplain {

class SigmoidLayer : public NetworkLayer {
public:
    explicit SigmoidLayer(Shape shape)
        : NetworkLayer{"sigmoid", shape, shape} {}

    Values computeLayerOutput(Values const & input) const override;
};

} // namespace spexplain

#endif // SPEXPLAIN_SIGMOID_LAYER_H

