#ifndef XAI_SMT_OPENSMTVERIFIER2_H
#define XAI_SMT_OPENSMTVERIFIER2_H

#include <verifiers/UnsatCoreVerifier.h>

#include <memory>

namespace opensmt {
struct PTRef;
class MainSolver;
} // namespace opensmt

namespace spexplain {
class Network2;
} // namespace spexplain

namespace xai::verifiers {

// Variant of OpenSMTVerifier that encodes a layer-based spexplain::Network2 (list of NetworkLayers)
// instead of the dense spexplain::Network. Each layer is encoded by a dedicated per-type encoder
// module, so the encoding scales to convolutional models. Supported layer types:
//   fc / cnn / add   affine maps (linear combination of the previous layer's signal)
//   relu             ReLU activation; the only layer creating neuron variables / layerSizes entries
//   maxpool          exact max encoding per pooling window
//   flatten          identity re-indexing
//   transpose        axis-permutation re-indexing
//   sigmoid          allowed only as the trailing layer, where it is dropped (monotone, so the
//                    encoded pre-activation logits match the .nnet networks)
// Any other layer type (or a sigmoid that is not trailing) triggers
// std::logic_error("Unimplemented! Unsupported layer type: ...").
//
// This is an intermediate implementation meant to co-exist with OpenSMTVerifier until Network2
// fully replaces Network.
class OpenSMTVerifier2 : public UnsatCoreVerifier {
public:
    OpenSMTVerifier2();
    virtual ~OpenSMTVerifier2();
    OpenSMTVerifier2(OpenSMTVerifier2 const &) = delete;
    OpenSMTVerifier2 & operator=(OpenSMTVerifier2 const &) = delete;
    OpenSMTVerifier2(OpenSMTVerifier2 &&) = default;
    OpenSMTVerifier2 & operator=(OpenSMTVerifier2 &&) = default;

    bool contains(::opensmt::PTRef const &, NodeIndex) const;

    std::size_t termSizeOf(::opensmt::PTRef const &) const;

    void assertSampleModel() override;

    void setUnsatCoreFilter(std::vector<NodeIndex> const &) override;

    void addTerm(::opensmt::PTRef const &);
    void addExplanationTerm(::opensmt::PTRef const &, std::string termNamePrefix = "");

    void addUpperBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;
    void addLowerBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;
    // Ensure that equalities and intervals correspond to just one assertion
    void addEquality(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;
    void addInterval(LayerIndex layer, NodeIndex var, Float lo, Float hi, bool explanationTerm = false) override;

    void addClassificationConstraint(NodeIndex node, Float threshold) override;

    void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, Float rhs) override;

    void addPreference(::opensmt::PTRef const &);

    void setTimeLimit(std::chrono::milliseconds) override;

    std::unique_ptr<spexplain::Explanation> getSampleModelRestrictions(spexplain::Framework const &) override;

    void resetSampleQuery() override;
    void resetSample() override;
    void resetSampleModel() override;

    UnsatCore getUnsatCore() const override;

    //+ remove from API
    opensmt::MainSolver const & getSolver() const;
    opensmt::MainSolver & getSolver();

    void printSmtLib2Query(std::ostream &) const override;

protected:
    // Initializes with the layer-based network representation.
    void initImpl(spexplain::Network2 const &) override;

    // Network2 has no dense-network view, so derive the default from the layer list instead.
    bool defaultEncodingReluLowerBounds() const override;

    void pushImpl() override;
    void popImpl() override;

    Answer checkImpl() override;

private:
    class OpenSMTImpl;
    std::unique_ptr<OpenSMTImpl> pimpl;
};

} // namespace xai::verifiers

#endif // XAI_SMT_OPENSMTVERIFIER2_H

