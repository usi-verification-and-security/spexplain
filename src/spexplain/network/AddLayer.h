#ifndef SPEXPLAIN_ADD_LAYER_H
#define SPEXPLAIN_ADD_LAYER_H

#include "NetworkLayer.h"

namespace spexplain {

class AddLayer : public NetworkLayer {
public:
    AddLayer(Shape shape, Values bias)
        : NetworkLayer{"add", shape, shape}, bias{std::move(bias)} {}

    Values computeLayerOutput(Values const & input) const override;

    Values const & getBias() const { return bias; }
    //TODO: Not sure
    // Bias contribution for output element `idx`, applying the same broadcasting rules as
    // computeLayerOutput (empty / size-1 / size==outputSize / size==outputShape.back() / cyclic).
    Float biasFor(std::size_t idx) const;

private:
    Values bias;
};

} // namespace spexplain

#endif // SPEXPLAIN_ADD_LAYER_H

