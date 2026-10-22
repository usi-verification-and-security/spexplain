#ifndef SPEXPLAIN_CNN_LAYER_H
#define SPEXPLAIN_CNN_LAYER_H

#include "NetworkLayer.h"

#include <array>
#include <cstddef>
#include <vector>

namespace spexplain {

/// 2-D convolution layer (ONNX Conv operator).
///
/// Shapes use the channel-first (NCHW) convention, but the batch dimension is
/// always omitted here because Network2 operates on single samples.
///
///   inputShape  = {inChannels, inputH, inputW}
///   filterShape = {outChannels, inChannels, filterH, filterW}
///   outputShape = {outChannels, outputH, outputW}
///
/// Filters and biases are stored in flat, row-major order as they come out of
/// the ONNX initializer (same memory layout as the ONNX TensorProto).
class CNNLayer : public NetworkLayer {
public:
    using Strides = std::array<std::size_t, 2>;   ///< {strideH, strideW}
    using Padding = std::array<std::size_t, 4>;   ///< {padTop, padBottom, padLeft, padRight}

    /// \param inputShape   {inC, H, W}
    /// \param filterShape  {outC, inC, kH, kW}  – must be consistent with inputShape
    /// \param filters      Flat weight tensor, ONNX layout [outC][inC][kH][kW]
    /// \param biases       One bias per output channel; may be empty (treated as 0)
    /// \param strides      {strideH, strideW}
    /// \param padding      {padTop, padBottom, padLeft, padRight}
    CNNLayer(Shape inputShape,
             Shape filterShape,
             Values filters,
             Values biases,
             Strides strides,
             Padding padding);

    Values computeLayerOutput(Values const & input) const override;

    Shape   const & getFilterShape() const { return filterShape; }
    Values  const & getFilters()     const { return filters; }
    Values  const & getBiases()      const { return biases; }
    Strides const & getStrides()     const { return strides; }
    Padding const & getPadding()     const { return padding; }

private:
    Shape   filterShape;   ///< {outC, inC, kH, kW}
    Values  filters;
    Values  biases;
    Strides strides;
    Padding padding;
};

} // namespace spexplain

#endif // SPEXPLAIN_CNN_LAYER_H

