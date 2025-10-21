
#include "ConvLayer.h"


namespace spexplain {
ConvLayer::Vector3D ConvLayer::padInput(const ConvLayer::Vector3D& input) const {
    Vector3D padded3D;
    for (auto const & channel : input) {
        int rows = channel.size();
        int cols = channel[0].size();
        Vector2D padded(rows + 2 * padding, Values(cols + 2 * padding, 0.0f));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                padded[i + padding][j + padding] = channel[i][j];
        padded3D.push_back(padded);
    }
    return padded3D;
}

ConvLayer::LayerOutputVariant ConvLayer::computeLayerOutput(LayerOutputVariant const & input) const {
    assert(std::holds_alternative<Vector3D>(input) or
           std::holds_alternative<Vector2D>(input));  //2D or 3D inputs
    // If 2D inputs, convert to 3D
    Vector3D input3D;
    if (std::holds_alternative<Vector2D>(input)) {
        auto const & input2D = std::get<Vector2D>(input);
        input3D.push_back(input2D);
    } else {
        auto &input3D = std::get<Vector3D>(input);
    }

    Vector3D inputPadded = input3D;
    if (padding>0) {
       inputPadded = padInput(input3D);
    }

    std::size_t inChannels = inputPadded.size();
    std::size_t inRows = inputPadded[0].size();
    std::size_t inCols = inputPadded[0][0].size();

    std::size_t outChannels = filters.size();
    std::size_t filterRows = filters[0][0].size();
    std::size_t filterCols = filters[0][0][0].size();

    std::size_t outRows = (inRows - filterRows) / stride + 1;
    std::size_t outCols = (inCols - filterCols) / stride + 1;

    Vector3D output(outChannels, Vector2D(outRows, Values(outCols, 0.0f)));

    for (std::size_t oc = 0; oc < outChannels; ++oc) {
        for (std::size_t i = 0; i < outRows; ++i) {
            for (std::size_t j = 0; j < outCols; ++j) {
                float sum = 0.0f;
                for (std::size_t ic = 0; ic < inChannels; ++ic) {
                    for (std::size_t m = 0; m < filterRows; ++m) {
                        for (std::size_t n = 0; n < filterCols; ++n) {
                            std::size_t x = i * stride + m;
                            std::size_t y = j * stride + n;
                            sum += inputPadded[ic][x][y] * filters[oc][ic][m][n];
                        }
                    }
                }
                float val = sum + bias[oc];
                if (followedByRelu && val < 0.0f) {
                    val = 0.0f;
                }
                output[oc][i][j] = val;
            }
        }
    }
    return output;
}

ConvLayer::WeightsVariant ConvLayer::getWeights(std::size_t kernelIndex) const {
    assert(kernelIndex < filters.size());
    return filters[kernelIndex];
}
}