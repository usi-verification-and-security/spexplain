#ifndef SPEXPLAIN_TRANSPOSE_LAYER_H
#define SPEXPLAIN_TRANSPOSE_LAYER_H

#include "NetworkLayer.h"

#include <cstddef>
#include <vector>

namespace spexplain {

class TransposeLayer : public NetworkLayer {
public:
    using Permutation = std::vector<std::size_t>;

    TransposeLayer(Shape inputShape, Permutation perm);

    Values computeLayerOutput(Values const & input) const override;

    // Mapping from each (flattened) output element to the (flattened) input element it reads,
    // i.e. output[o] == input[outputToInputIndexMap()[o]].
    std::vector<std::size_t> outputToInputIndexMap() const;

private:
    static Shape permutedShape(Shape const & inputShape, Permutation const & perm);

    Permutation perm;
};

} // namespace spexplain

#endif // SPEXPLAIN_TRANSPOSE_LAYER_H

