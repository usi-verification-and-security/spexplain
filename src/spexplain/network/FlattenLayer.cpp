//
// Created by labbaf on 05.09.2025.
//

#include "FlattenLayer.h"

namespace spexplain {
NetworkLayer::LayerOutputVariant FlattenLayer::computeLayerOutput(LayerOutputVariant const & input) const {
    // Flatten the input
    if (std::holds_alternative<Values>(input)) {
        return input;
    } else if (std::holds_alternative<Vector2D>(input)) {
        Vector2D const & input2D = std::get<Vector2D>(input);
        Values output;
        for (auto const & row : input2D) {
            output.insert(output.end(), row.begin(), row.end());
        }
        return output;
    } else if (std::holds_alternative<Vector3D>(input)) {
        Vector3D const & input3D = std::get<Vector3D>(input);
        Values output;
        for (auto const & matrix : input3D) {
            for (auto const & row : matrix) {
                output.insert(output.end(), row.begin(), row.end());
            }
        }
        return output;
    } else {
        throw std::logic_error("Unsupported input type for Flatten layer");
    }
}



}