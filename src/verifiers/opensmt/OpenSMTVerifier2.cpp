#include "OpenSMTVerifier2.h"

#include <api/MainSolver.h>
#include <common/StringConv.h>
#include <logics/ArithLogic.h>
#include <logics/LogicFactory.h>

#include <spexplain/common/Macro.h>
#include <spexplain/framework/explanation/ConjunctExplanation.h>
#include <spexplain/framework/explanation/opensmt/FormulaExplanation.h>
#include <spexplain/network/AddLayer.h>
#include <spexplain/network/CNNLayer.h>
#include <spexplain/network/FCLayer.h>
#include <spexplain/network/FlattenLayer.h>
#include <spexplain/network/MaxPoolLayer.h>
#include <spexplain/network/Network2.h>
#include <spexplain/network/ReLULayer.h>
#include <spexplain/network/SigmoidLayer.h>
#include <spexplain/network/TransposeLayer.h>

#include <algorithm>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xai::verifiers {

using namespace opensmt;

namespace { // Helper methods
    FastRational floatToRational(Float value);

    // One output element of an affine map: out = bias + sum_k coeffs[k].second * in[coeffs[k].first].
    struct AffineRow {
        Float bias{};
        std::vector<std::pair<std::size_t, Float>> coeffs; // (input index, weight)
    };
    using AffineRows = std::vector<AffineRow>;

    AffineRows fcRows(spexplain::FCLayer const &);
    AffineRows cnnRows(spexplain::CNNLayer const &);
    AffineRows addRows(spexplain::AddLayer const &);
} // namespace

class OpenSMTVerifier2::OpenSMTImpl {
public:
    OpenSMTImpl(OpenSMTVerifier2 & verifier_) : verifier{verifier_} {}

    bool contains(PTRef const &, NodeIndex) const;

    std::size_t termSizeOf(PTRef const &) const;

    void assertSampleModel();

    void setUnsatCoreFilter(std::vector<NodeIndex> const &);

    void addTerm(PTRef const &);
    void addExplanationTerm(PTRef const &, std::string termNamePrefix = "");

    PTRef makeUpperBound(LayerIndex layer, NodeIndex node, Float value) {
        return makeUpperBound(layer, node, floatToRational(value));
    }
    PTRef makeLowerBound(LayerIndex layer, NodeIndex node, Float value) {
        return makeLowerBound(layer, node, floatToRational(value));
    }
    PTRef makeEquality(LayerIndex layer, NodeIndex node, Float value) {
        return makeEquality(layer, node, floatToRational(value));
    }
    PTRef makeInterval(LayerIndex layer, NodeIndex node, Float lo, Float hi) {
        return makeInterval(layer, node, floatToRational(lo), floatToRational(hi));
    }
    PTRef makeUpperBound(LayerIndex layer, NodeIndex node, FastRational value);
    PTRef makeLowerBound(LayerIndex layer, NodeIndex node, FastRational value);
    PTRef makeEquality(LayerIndex layer, NodeIndex node, FastRational value) {
        FastRational valueCp = value;
        return makeInterval(layer, node, std::move(value), std::move(valueCp));
    }
    PTRef makeInterval(LayerIndex layer, NodeIndex node, FastRational lo, FastRational hi);

    PTRef addUpperBound(LayerIndex layer, NodeIndex node, Float value, bool explanationTerm = false);
    PTRef addLowerBound(LayerIndex layer, NodeIndex node, Float value, bool explanationTerm = false);
    PTRef addEquality(LayerIndex layer, NodeIndex node, Float value, bool explanationTerm = false);
    PTRef addInterval(LayerIndex layer, NodeIndex node, Float lo, Float hi, bool explanationTerm = false);

    void addClassificationConstraint(NodeIndex node, Float threshold);

    void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, Float rhs);

    void addPreference(PTRef const &);

    void init();

    void push();
    void pop();

    void setTimeLimit(std::chrono::milliseconds);

    Answer check();

    void resetSampleQuery();
    void resetSample();
    void resetSampleModel();

    std::unique_ptr<spexplain::Explanation> getSampleModelRestrictions(spexplain::Framework const &);

    UnsatCore getUnsatCore() const;

    opensmt::MainSolver const & getSolver() const { return *solver; }
    opensmt::MainSolver & getSolver() { return *solver; }

    void printSmtLib2Query(std::ostream &) const;

private:
    //! sync with the framework
    static std::string inputVarName(NodeIndex node) { return "x" + std::to_string(node + 1); }

    static std::string neuronVarName(LayerIndex layer, NodeIndex node, std::string base_layer_name = "l", std::string basename = "n") {
        return std::move(base_layer_name) + std::to_string(layer) + "_" + std::move(basename) + std::to_string(node + 1);
    }

    std::string makeExplanationTermName(std::string prefix = "") {
        return prefix + "t" + std::to_string(explanationTerms.size() - 1);
    }

    bool storingNeuronTerms() const {
        if (not verifier.encodingNeuronVars()) { return true; }
        if (not verifier.hasFixedNeuronActivations()) { return false; }
        return not verifier.allowedNeuronVarsInExplanations();
    }

    bool containsInputLowerBound(PTRef term) const { return inputVarLowerBoundToIndex.contains(term); }
    bool containsInputUpperBound(PTRef term) const { return inputVarUpperBoundToIndex.contains(term); }
    bool containsInputEquality(PTRef term) const { return inputVarEqualityToIndex.contains(term); }
    bool containsInputInterval(PTRef term) const { return inputVarIntervalToIndex.contains(term); }
    NodeIndex nodeIndexOfInputLowerBound(PTRef term) const { return inputVarLowerBoundToIndex.at(term); }
    NodeIndex nodeIndexOfInputUpperBound(PTRef term) const { return inputVarUpperBoundToIndex.at(term); }
    NodeIndex nodeIndexOfInputEquality(PTRef term) const { return inputVarEqualityToIndex.at(term); }
    NodeIndex nodeIndexOfInputInterval(PTRef term) const { return inputVarIntervalToIndex.at(term); }

    PTRef encodeNeuron(LayerIndex layer, NodeIndex node, PTRef neuronVar, PTRef varInput, PTRef termInput);

    // The (flattened) tensor flowing between two layers. Two parallel streams are maintained,
    // mirroring the dual tracking of the previous implementation:
    //   vars  - live when encodingNeuronVars(); built on top of neuron variables
    //   terms - live when storingNeuronTerms();  built as self-contained ite terms
    struct LayerSignal {
        std::vector<PTRef> vars;
        std::vector<PTRef> terms;

        bool hasVars() const { return not vars.empty(); }
        bool hasTerms() const { return not terms.empty(); }
        std::size_t size() const { return hasVars() ? vars.size() : terms.size(); }
    };

    // One module per layer type. Encodes a single layer, asserting whatever constraints it needs,
    // and returns the signal consumed by the next layer. Nested inside OpenSMTImpl so that it may
    // access the enclosing instance's private members through the explicit `impl` parameter.
    class LayerEncoder {
    public:
        virtual ~LayerEncoder() = default;
        virtual LayerSignal encode(OpenSMTImpl & impl, LayerSignal const & in) const = 0;
    };

    // fc / cnn / add: affine maps expressed as sparse rows. Creates NO variables and does not touch
    // layerSizes / neuronVars; it only materialises the linear combinations as plus-terms.
    class AffineEncoder : public LayerEncoder {
    public:
        explicit AffineEncoder(AffineRows rows) : rows{std::move(rows)} {}
        LayerSignal encode(OpenSMTImpl & impl, LayerSignal const & in) const override;

    private:
        AffineRows rows;
    };

    // relu: the only encoder that creates neuron variables and appends to layerSizes / neuronVars /
    // neuronTerms. Reuses encodeNeuron for the full ReLU encoding.
    class ReLUEncoder : public LayerEncoder {
    public:
        LayerSignal encode(OpenSMTImpl & impl, LayerSignal const & in) const override;
    };

    // flatten / transpose: pure re-indexings, out[o] = in[outToIn[o]]. No constraints, no variables.
    class IndexMapEncoder : public LayerEncoder {
    public:
        explicit IndexMapEncoder(std::vector<std::size_t> outToIn) : outToIn{std::move(outToIn)} {}
        LayerSignal encode(OpenSMTImpl & impl, LayerSignal const & in) const override;

    private:
        std::vector<std::size_t> outToIn;
    };

    // maxpool: exact max encoding per output window.
    class MaxPoolEncoder : public LayerEncoder {
    public:
        explicit MaxPoolEncoder(std::vector<std::vector<std::size_t>> windows) : windows{std::move(windows)} {}
        LayerSignal encode(OpenSMTImpl & impl, LayerSignal const & in) const override;

    private:
        std::vector<std::vector<std::size_t>> windows;
    };

    static std::unique_ptr<LayerEncoder> makeLayerEncoder(spexplain::NetworkLayer const &);

    OpenSMTVerifier2 & verifier;

    std::unique_ptr<ArithLogic> logic;
    std::unique_ptr<MainSolver> solver;
    std::unique_ptr<SMTConfig> config;

    std::vector<PTRef> inputVars;
    std::vector<std::vector<PTRef>> neuronVars;
    std::vector<std::vector<PTRef>> neuronTerms;
    std::vector<PTRef> outputVarsOrTerms;
    std::vector<std::size_t> layerSizes;

    // Index of the next ReLU hidden layer; shared across encoder invocations of a single
    // assertSampleModel run. Reset to 1 at the start of the run (layer 0 is the input layer).
    LayerIndex nextHiddenLayerIndex{1};
    // Distinct counter for maxpool auxiliary variables, so their names cannot collide with neuronVarName.
    std::size_t nextPoolIndex{0};

    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarsMap;
    std::unordered_map<PTRef, std::pair<LayerIndex, NodeIndex>, PTRefHash> neuronVarsMap;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> outputVarsMap;

    std::vector<PTRef> sampleModelRestrictions;

    std::vector<NodeIndex> unsatCoreNodeFilter;

    std::vector<PTRef> explanationTerms;
    std::unordered_map<PTRef, std::size_t, PTRefHash> unsatCoreNonFilteredTermsToIndex;
    std::unordered_map<PTRef, std::size_t, PTRefHash> unsatCoreFilteredTermsToIndex;

    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarLowerBoundToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarUpperBoundToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarEqualityToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarIntervalToIndex;
};

OpenSMTVerifier2::OpenSMTVerifier2() : pimpl{std::make_unique<OpenSMTImpl>(*this)} {}

OpenSMTVerifier2::~OpenSMTVerifier2() {}

bool OpenSMTVerifier2::contains(PTRef const & term, NodeIndex node) const {
    return pimpl->contains(term, node);
}

std::size_t OpenSMTVerifier2::termSizeOf(PTRef const & term) const {
    return pimpl->termSizeOf(term);
}

void OpenSMTVerifier2::assertSampleModel() {
    pimpl->assertSampleModel();
}

void OpenSMTVerifier2::setUnsatCoreFilter(std::vector<NodeIndex> const & filter) {
    pimpl->setUnsatCoreFilter(filter);
}

void OpenSMTVerifier2::addTerm(PTRef const & term) {
    pimpl->addTerm(term);
}

void OpenSMTVerifier2::addExplanationTerm(PTRef const & term, std::string termNamePrefix) {
    pimpl->addExplanationTerm(term, std::move(termNamePrefix));
}

void OpenSMTVerifier2::addUpperBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm) {
    pimpl->addUpperBound(layer, var, value, explanationTerm);
}

void OpenSMTVerifier2::addLowerBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm) {
    pimpl->addLowerBound(layer, var, value, explanationTerm);
}

void OpenSMTVerifier2::addEquality(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm) {
    pimpl->addEquality(layer, var, value, explanationTerm);
}

void OpenSMTVerifier2::addInterval(LayerIndex layer, NodeIndex var, Float lo, Float hi, bool explanationTerm) {
    pimpl->addInterval(layer, var, lo, hi, explanationTerm);
}

void OpenSMTVerifier2::addClassificationConstraint(NodeIndex node, Float threshold = 0) {
    pimpl->addClassificationConstraint(node, threshold);
}

void OpenSMTVerifier2::addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, Float rhs) {
    pimpl->addConstraint(layer, lhs, rhs);
}

void OpenSMTVerifier2::addPreference(PTRef const & term) {
    pimpl->addPreference(term);
}

void OpenSMTVerifier2::initImpl(spexplain::Network2 const & nw) {
    Verifier::initImpl(nw);
    pimpl->init();
}

bool OpenSMTVerifier2::defaultEncodingReluLowerBounds() const {
    if (not encodingNeuronVars()) { return false; }

    auto const & network = getNetwork2();
    std::size_t nHiddenLayers = 0;
    for (auto const & layer : network.getLayers()) {
        if (layer and layer->getType() == "relu") { ++nHiddenLayers; }
    }
    if (nHiddenLayers > 1) { return false; }

    return true;
}

void OpenSMTVerifier2::pushImpl() {
    pimpl->push();
}

void OpenSMTVerifier2::popImpl() {
    pimpl->pop();
}

void OpenSMTVerifier2::setTimeLimit(std::chrono::milliseconds limit) {
    pimpl->setTimeLimit(limit);
}

Verifier::Answer OpenSMTVerifier2::checkImpl() {
    return pimpl->check();
}

void OpenSMTVerifier2::resetSampleQuery() {
    pimpl->resetSampleQuery();
    UnsatCoreVerifier::resetSampleQuery();
}

void OpenSMTVerifier2::resetSample() {
    pimpl->resetSample();
    UnsatCoreVerifier::resetSample();
}

void OpenSMTVerifier2::resetSampleModel() {
    pimpl->resetSampleModel();
    UnsatCoreVerifier::resetSampleModel();
}

std::unique_ptr<spexplain::Explanation> OpenSMTVerifier2::getSampleModelRestrictions(spexplain::Framework const & fw) {
    return pimpl->getSampleModelRestrictions(fw);
}

UnsatCore OpenSMTVerifier2::getUnsatCore() const {
    return pimpl->getUnsatCore();
}

opensmt::MainSolver const & OpenSMTVerifier2::getSolver() const {
    return pimpl->getSolver();
}

opensmt::MainSolver & OpenSMTVerifier2::getSolver() {
    return pimpl->getSolver();
}

void OpenSMTVerifier2::printSmtLib2Query(std::ostream & os) const {
    return pimpl->printSmtLib2Query(os);
}

/*
 * Actual implementation
 */

namespace { // Helper methods
    FastRational floatToRational(Float value) {
        auto s = std::to_string(value);
        char * rationalString;
        opensmt::stringToRational(rationalString, s.c_str());
        auto res = FastRational(rationalString);
        free(rationalString);
        return res;
    }

    Verifier::Answer toAnswer(sstat res) {
        if (res == s_False) { return Verifier::Answer::UNSAT; }
        if (res == s_True) { return Verifier::Answer::SAT; }
        if (res == s_Error) { return Verifier::Answer::ERROR; }
        if (res == s_Undef) { return Verifier::Answer::UNKNOWN; }
        return Verifier::Answer::UNKNOWN;
    }

    // Mirror FCLayer::computeLayerOutput: weights are stored flat as weights[in * outputSize + out],
    // bias = biases.empty() ? 0 : biases[out % biases.size()]. Every weight (including zeros) is kept.
    AffineRows fcRows(spexplain::FCLayer const & fc) {
        auto const & weights = fc.getWeights();
        auto const & biases = fc.getBiases();
        std::size_t const inputSize = fc.getInputSize();
        std::size_t const outputSize = fc.getOutputSize();
        assert(weights.size() == inputSize * outputSize);

        AffineRows rows(outputSize);
        for (std::size_t out = 0; out < outputSize; ++out) {
            AffineRow & row = rows[out];
            row.bias = biases.empty() ? Float{} : biases[out % biases.size()];
            row.coeffs.reserve(inputSize);
            for (std::size_t in = 0; in < inputSize; ++in) {
                row.coeffs.emplace_back(in, weights[in * outputSize + out]);
            }
        }
        return rows;
    }

    // Mirror CNNLayer::computeLayerOutput (NCHW, batch dim omitted).
    AffineRows cnnRows(spexplain::CNNLayer const & cnn) {
        auto const & inputShape = cnn.getInputShape();
        auto const & filterShape = cnn.getFilterShape();
        auto const & outputShape = cnn.getOutputShape();
        auto const & filters = cnn.getFilters();
        auto const & biases = cnn.getBiases();
        auto const & strides = cnn.getStrides();
        auto const & padding = cnn.getPadding();

        std::size_t const inC = inputShape[0];
        std::size_t const inH = inputShape[1];
        std::size_t const inW = inputShape[2];

        std::size_t const outC = filterShape[0];
        std::size_t const kH = filterShape[2];
        std::size_t const kW = filterShape[3];

        std::size_t const outH = outputShape[1];
        std::size_t const outW = outputShape[2];

        std::size_t const sH = strides[0];
        std::size_t const sW = strides[1];
        std::size_t const padT = padding[0];
        std::size_t const padL = padding[2];

        AffineRows rows(outC * outH * outW);
        for (std::size_t oc = 0; oc < outC; ++oc) {
            Float const bias = biases.empty() ? Float{} : biases[oc];
            for (std::size_t oh = 0; oh < outH; ++oh) {
                for (std::size_t ow = 0; ow < outW; ++ow) {
                    AffineRow & row = rows[oc * outH * outW + oh * outW + ow];
                    row.bias = bias;
                    for (std::size_t ic = 0; ic < inC; ++ic) {
                        for (std::size_t kh = 0; kh < kH; ++kh) {
                            for (std::size_t kw = 0; kw < kW; ++kw) {
                                std::size_t const ih = oh * sH + kh;
                                std::size_t const iw = ow * sW + kw;
                                if (ih < padT || iw < padL) { continue; }
                                std::size_t const realH = ih - padT;
                                std::size_t const realW = iw - padL;
                                if (realH >= inH || realW >= inW) { continue; }

                                std::size_t const inIdx = ic * inH * inW + realH * inW + realW;
                                Float const w = filters[oc * inC * kH * kW + ic * kH * kW + kh * kW + kw];
                                row.coeffs.emplace_back(inIdx, w);
                            }
                        }
                    }
                }
            }
        }
        return rows;
    }

    // Mirror AddLayer::computeLayerOutput: out[i] = in[i] + biasFor(i).
    AffineRows addRows(spexplain::AddLayer const & add) {
        std::size_t const outputSize = add.getOutputSize();
        AffineRows rows(outputSize);
        for (std::size_t i = 0; i < outputSize; ++i) {
            rows[i].bias = add.biasFor(i);
            rows[i].coeffs.emplace_back(i, Float{1});
        }
        return rows;
    }
} // namespace

bool OpenSMTVerifier2::OpenSMTImpl::contains(PTRef const & term, NodeIndex node) const {
    auto & inputVar = inputVars.at(node);
    return logic->contains(term, inputVar);
}

std::size_t OpenSMTVerifier2::OpenSMTImpl::termSizeOf(PTRef const & term) const {
    Pterm const & pterm = logic->getPterm(term);

    if (logic->isAtom(term)) { return 1; }

    if (logic->isNot(term)) {
        assert(pterm.size() == 1);
        auto & negTerm = *pterm.begin();
        assert(not logic->isNot(negTerm));
        // just care about no. literals, so ignore the negations themselves
        return termSizeOf(negTerm);
    }

    assert(logic->isAnd(term) or logic->isOr(term));
    std::size_t totalSize{};
    for (PTRef const & argTerm : pterm) {
        totalSize += termSizeOf(argTerm);
    }
    return totalSize;
}

OpenSMTVerifier2::OpenSMTImpl::LayerSignal OpenSMTVerifier2::OpenSMTImpl::AffineEncoder::encode(OpenSMTImpl & impl, LayerSignal const & in) const {
    auto & logic = *impl.logic;

    auto emit = [&logic](AffineRow const & row, std::vector<PTRef> const & stream) {
        std::vector<PTRef> addends;
        addends.reserve(row.coeffs.size() + 1);
        addends.push_back(logic.mkRealConst(floatToRational(row.bias)));
        for (auto const & [idx, weight] : row.coeffs) {
            PTRef weightTerm = logic.mkRealConst(floatToRational(weight));
            addends.push_back(logic.mkTimes(weightTerm, stream[idx]));
        }
        return logic.mkPlus(addends);
    };

    LayerSignal out;
    if (in.hasVars()) {
        out.vars.reserve(rows.size());
        for (auto const & row : rows) { out.vars.push_back(emit(row, in.vars)); }
    }
    if (in.hasTerms()) {
        out.terms.reserve(rows.size());
        for (auto const & row : rows) { out.terms.push_back(emit(row, in.terms)); }
    }
    return out;
}

//TODO:Faezeh check - Check the ReLU encoding if it contains the guiding and fixing procedures!
OpenSMTVerifier2::OpenSMTImpl::LayerSignal OpenSMTVerifier2::OpenSMTImpl::ReLUEncoder::encode(OpenSMTImpl & impl, LayerSignal const & in) const {
    bool const encodeNeuronVars = impl.verifier.encodingNeuronVars();
    bool const storeNeuronTerms = impl.storingNeuronTerms();

    std::size_t const layerSize = in.size();
    LayerIndex const hiddenLayer = impl.nextHiddenLayerIndex;

    std::vector<PTRef> currentLayerVars;
    std::vector<PTRef> currentLayerTerms;
    if (encodeNeuronVars) { currentLayerVars.reserve(layerSize); }
    if (storeNeuronTerms) { currentLayerTerms.reserve(layerSize); }

    for (NodeIndex node = 0u; node < layerSize; ++node) {
        PTRef const varInput = encodeNeuronVars ? in.vars[node] : PTRef_Undef;
        PTRef const termInput = storeNeuronTerms ? in.terms[node] : PTRef_Undef;

        PTRef neuronVar = PTRef_Undef;
        if (encodeNeuronVars) {
            auto neuronName = neuronVarName(hiddenLayer, node, "l", "n");
            neuronVar = impl.logic->mkRealVar(neuronName.c_str());
            currentLayerVars.push_back(neuronVar);
            auto [it, inserted] = impl.neuronVarsMap.emplace(neuronVar, std::make_pair(hiddenLayer, node));
            assert(inserted);
            (void)it;
        }

        [[maybe_unused]]
        PTRef neuronTerm = impl.encodeNeuron(hiddenLayer, node, neuronVar, varInput, termInput);
        assert(storeNeuronTerms == (neuronTerm != PTRef_Undef));
        if (not storeNeuronTerms) { continue; }
        assert(not impl.logic->isVar(neuronTerm));
        assert(impl.logic->getSortRef(neuronTerm) != impl.logic->getSort_bool());
        currentLayerTerms.push_back(neuronTerm);
    }

    impl.layerSizes.push_back(layerSize);

    LayerSignal out;
    assert(encodeNeuronVars or currentLayerVars.empty());
    if (encodeNeuronVars) {
        impl.neuronVars.push_back(currentLayerVars);
        out.vars = std::move(currentLayerVars);
    }
    assert(storeNeuronTerms or currentLayerTerms.empty());
    if (storeNeuronTerms) {
        impl.neuronTerms.push_back(currentLayerTerms);
        out.terms = std::move(currentLayerTerms);
    }

    ++impl.nextHiddenLayerIndex;
    return out;
}

//TODO:Faezeh - Check - Is it only usefull for reverse? is it working fine for the reversed 2D matrixes?
OpenSMTVerifier2::OpenSMTImpl::LayerSignal OpenSMTVerifier2::OpenSMTImpl::IndexMapEncoder::encode([[maybe_unused]] OpenSMTImpl & impl,
                                                                   LayerSignal const & in) const {
    LayerSignal out;
    if (in.hasVars()) {
        out.vars.reserve(outToIn.size());
        for (std::size_t idx : outToIn) { out.vars.push_back(in.vars[idx]); }
    }
    if (in.hasTerms()) {
        out.terms.reserve(outToIn.size());
        for (std::size_t idx : outToIn) { out.terms.push_back(in.terms[idx]); }
    }
    return out;
}

//TODO:Faezeh - Check - How the MaxPool work?
//TODO:Faezeh - Also write a MinPool
OpenSMTVerifier2::OpenSMTImpl::LayerSignal OpenSMTVerifier2::OpenSMTImpl::MaxPoolEncoder::encode(OpenSMTImpl & impl, LayerSignal const & in) const {
    auto & logic = *impl.logic;

    bool const encodeNeuronVars = impl.verifier.encodingNeuronVars();
    bool const storeNeuronTerms = impl.storingNeuronTerms();

    std::size_t const poolIdx = impl.nextPoolIndex++;

    LayerSignal out;
    if (encodeNeuronVars) { out.vars.reserve(windows.size()); }
    if (storeNeuronTerms) { out.terms.reserve(windows.size()); }

    for (NodeIndex node = 0u; node < windows.size(); ++node) {
        auto const & window = windows[node];
        if (window.empty()) { throw std::logic_error("Unimplemented! Empty max-pool window"); }

        if (encodeNeuronVars) {
            // Fresh auxiliary variable m with m >= a_j for every input a_j and Or_j (m == a_j).
            // The name is deliberately distinct from neuronVarName so it cannot collide, and the
            // variable is NOT registered in neuronVarsMap nor does it push a layerSizes entry: the
            // fixed/preferred-activation machinery and the layerSizes layout are ReLU-specific, and
            // adding entries would silently shift the output layer index Expand::assertClassification
            // computes.
            // auto name = "p" + std::to_string(poolIdx) + "_m" + std::to_string(node + 1);
            auto name = neuronVarName(poolIdx, node, "p", "m");

            PTRef m = logic.mkRealVar(name.c_str());

            std::vector<PTRef> eqs;
            eqs.reserve(window.size());
            for (std::size_t idx : window) {
                PTRef const a = in.vars[idx];
                impl.addTerm(logic.mkGeq(m, a));
                eqs.push_back(logic.mkEq(m, a));
            }
            impl.addTerm(logic.mkOr(eqs));
            out.vars.push_back(m);
        }

        if (storeNeuronTerms) {
            PTRef acc = in.terms[window[0]];
            for (std::size_t j = 1; j < window.size(); ++j) {
                PTRef const a = in.terms[window[j]];
                acc = logic.mkIte(logic.mkGeq(acc, a), acc, a);
            }
            out.terms.push_back(acc);
        }
    }

    return out;
}

std::unique_ptr<OpenSMTVerifier2::OpenSMTImpl::LayerEncoder>
OpenSMTVerifier2::OpenSMTImpl::makeLayerEncoder(spexplain::NetworkLayer const & layer) {
    std::string const & type = layer.getType();

    if (type == "fc") {
        return std::make_unique<AffineEncoder>(fcRows(dynamic_cast<spexplain::FCLayer const &>(layer)));
    }
    if (type == "cnn") {
        return std::make_unique<AffineEncoder>(cnnRows(dynamic_cast<spexplain::CNNLayer const &>(layer)));
    }
    if (type == "add") {
        return std::make_unique<AffineEncoder>(addRows(dynamic_cast<spexplain::AddLayer const &>(layer)));
    }
    if (type == "relu") { return std::make_unique<ReLUEncoder>(); }
    if (type == "flatten") {
        // Flatten is a no-op re-indexing (identity map).
        // TODO:Faezeh - just need to make sure it is flattened the same way.
        std::vector<std::size_t> identity(layer.getOutputSize());
        for (std::size_t i = 0; i < identity.size(); ++i) { identity[i] = i; }
        return std::make_unique<IndexMapEncoder>(std::move(identity));
    }
    if (type == "transpose") {
        return std::make_unique<IndexMapEncoder>(
            dynamic_cast<spexplain::TransposeLayer const &>(layer).outputToInputIndexMap());
    }
    if (type == "maxpool") {
        return std::make_unique<MaxPoolEncoder>(dynamic_cast<spexplain::MaxPoolLayer const &>(layer).windowIndices());
    }
    if (type == "sigmoid") {
        // A sigmoid is not encodable in linear arithmetic. A *trailing* sigmoid is monotone and is
        // therefore dropped (see Network2::nEffectiveLayers), which is the default; reaching this
        // point means the sigmoid is either not trailing or dropping was disabled.
        throw std::logic_error("Unimplemented! Unsupported layer type: sigmoid (only a trailing sigmoid is "
                               "supported, and only with --drop-sigmoid true)");
    }

    throw std::logic_error("Unimplemented! Unsupported layer type: " + type);
}

//+ move some common parts into assertGroundModel, but incremental assertions seem much slower
//TODO:Faezeh - compare the function with the old OpenSMTVerifier
void OpenSMTVerifier2::OpenSMTImpl::assertSampleModel() {
    auto const & network = verifier.getNetwork2();
    auto const & layers = network.getLayers();

    bool const encodeNeuronVars = verifier.encodingNeuronVars();
    bool const encodeOutputVars = verifier.encodingOutputVars();
    bool const storeNeuronTerms = storingNeuronTerms();

    // Reset per-run state shared across encoder invocations.
    layerSizes.clear();
    nextHiddenLayerIndex = 1; // layer 0 is the input layer
    nextPoolIndex = 0;

    // Determine the (flattened) size of the input tensor.
    std::size_t inputSize = 0;
    for (auto const & shape = network.getInputShape(); auto dim : shape) {
        inputSize = (inputSize == 0 ? 1 : inputSize) * dim;
    }
    if (inputSize == 0 and not layers.empty()) { inputSize = layers.front()->getInputSize(); }

    // Store information about layer sizes; layerSizes[0] is the input layer, the last is the output layer.
    layerSizes.push_back(inputSize);

    // create input variables
    for (NodeIndex i = 0u; i < inputSize; ++i) {
        auto name = inputVarName(i);
        PTRef var = logic->mkRealVar(name.c_str());
        inputVars.push_back(var);
        auto [it, inserted] = inputVarsMap.emplace(var, i);
        assert(inserted);
        (void)it;
    }

    // Collect hard bounds on inputs
    std::vector<PTRef> bounds;
    bounds.reserve(2 * inputVars.size());
    for (NodeIndex i = 0; i < inputVars.size(); ++i) {
        Float lb = network.getInputLowerBound(i);
        Float ub = network.getInputUpperBound(i);
        bounds.push_back(logic->mkGeq(inputVars[i], logic->mkRealConst(floatToRational(lb))));
        bounds.push_back(logic->mkLeq(inputVars[i], logic->mkRealConst(floatToRational(ub))));
    }
    addTerm(logic->mkAnd(bounds));

    // A trailing sigmoid is monotone and therefore irrelevant for the encoded (pre-activation)
    // output; it is dropped so that the encoding matches the logit semantics of the NNet networks.
    // Network2 applies the very same rule when evaluating, so both agree on the network output.
    std::size_t const end = network.nEffectiveLayers();

    std::size_t nReluLayers = 0;
    for (std::size_t li = 0; li < end; ++li) {
        if (not layers[li]) { throw std::logic_error("Unimplemented! Null layer in Network2"); }
        if (layers[li]->getType() == "relu") { ++nReluLayers; }
    }

    // Seed neuronVars/neuronTerms with the empty input-layer placeholder.
    if (encodeNeuronVars) {
        neuronVars.reserve(nReluLayers + 1);
        neuronVars.emplace_back();
    }
    if (storeNeuronTerms) {
        neuronTerms.reserve(nReluLayers + 1);
        neuronTerms.emplace_back();
    }

    // Feed the input signal through the per-layer encoders.
    LayerSignal signal;
    if (encodeNeuronVars) { signal.vars = inputVars; }
    if (storeNeuronTerms) { signal.terms = inputVars; }

    for (std::size_t li = 0; li < end; ++li) {
        signal = makeLayerEncoder(*layers[li])->encode(*this, signal);
    }

    // Materialise the output layer from the final signal.
    std::size_t const outputSize = signal.size();
    outputVarsOrTerms.clear();
    for (NodeIndex node = 0u; node < outputSize; ++node) {
        PTRef const value = signal.hasVars() ? signal.vars[node] : signal.terms[node];

        if (encodeOutputVars) {
            auto name = "o" + std::to_string(node + 1);
            PTRef var = logic->mkRealVar(name.c_str());
            outputVarsOrTerms.push_back(var);
            auto [it, inserted] = outputVarsMap.emplace(var, node);
            assert(inserted);
            (void)it;

            addTerm(logic->mkEq(var, value));
        } else {
            outputVarsOrTerms.push_back(value);
        }
    }

    layerSizes.push_back(outputSize);
}

PTRef OpenSMTVerifier2::OpenSMTImpl::encodeNeuron(LayerIndex layer, NodeIndex node, PTRef neuronVar, PTRef varInput,
                                                  PTRef termInput) {
    assert(layer > 0);

    // Construct lazily on demand, input may be large
    static auto const activeCondF = [](ArithLogic & logic_, PTRef input, PTRef zero) {
        return logic_.mkGeq(input, zero);
    };
    static auto const inactiveCondF = [](ArithLogic & logic_, PTRef input, PTRef zero) {
        return logic_.mkNot(activeCondF(logic_, input, zero));
    };

    static auto const activeEqF = [](ArithLogic & logic_, PTRef neuronVar_, PTRef varInput) {
        return logic_.mkEq(neuronVar_, varInput);
    };
    static auto const inactiveEqF = [](ArithLogic & logic_, PTRef neuronVar_, PTRef zero) {
        return logic_.mkEq(neuronVar_, zero);
    };

    static auto const iteF = [](ArithLogic & logic_, PTRef activeCond, PTRef input, PTRef zero) {
        return logic_.mkIte(activeCond, input, zero);
    };

    bool const encodeNeuronVars = verifier.encodingNeuronVars();
    bool const encodeReluLowerBounds = verifier.encodingReluLowerBounds();
    bool const storeNeuronTerms = storingNeuronTerms();
    assert(encodeNeuronVars or storeNeuronTerms);

    assert(encodeNeuronVars == (neuronVar != PTRef_Undef));
    assert(encodeNeuronVars == (varInput != PTRef_Undef));
    assert(storeNeuronTerms == (termInput != PTRef_Undef));

    PTRef zero = logic->getTerm_RealZero();

    PTRef condInput = storeNeuronTerms ? termInput : varInput;

    if (auto optFixedActivation = verifier.getFixedNeuronActivation(layer, node)) {
        PTRef cond;
        PTRef neuronTerm;
        if (*optFixedActivation) {
            if (encodeNeuronVars) { addTerm(activeEqF(*logic, neuronVar, varInput)); }
            cond = activeCondF(*logic, condInput, zero);
            assert(not storeNeuronTerms or condInput == termInput);
            neuronTerm = storeNeuronTerms ? termInput : PTRef_Undef;
        } else {
            if (encodeNeuronVars) { addTerm(inactiveEqF(*logic, neuronVar, zero)); }
            cond = inactiveCondF(*logic, condInput, zero);
            neuronTerm = storeNeuronTerms ? zero : PTRef_Undef;
        }

        // This can happen if there is a mismatch between float and real computation of the activation
        // We only detect the cases when it is simplified to false
        if (logic->isFalse(cond)) {
            throw std::runtime_error{"Attempt to fix "s + (*optFixedActivation ? "active" : "inactive") +
                                     " neuron condition that is false: layer=" + std::to_string(layer) +
                                     " node=" + std::to_string(node)};
        }
        if (not logic->isTrue(cond)) { sampleModelRestrictions.push_back(cond); }

        return neuronTerm;
    }

    PTRef activeCond;
    PTRef inactiveCond;
    if (storeNeuronTerms or not(encodeNeuronVars and encodeReluLowerBounds)) {
        activeCond = activeCondF(*logic, condInput, zero);
        inactiveCond = inactiveCondF(*logic, condInput, zero);
    }

    PTRef activeLeq;
    PTRef inactiveLeq;
    if (encodeNeuronVars) {
        if (encodeReluLowerBounds) {
            // Hard constraints
            PTRef activeGeq = logic->mkGeq(neuronVar, varInput);
            PTRef inactiveGeq = logic->mkGeq(neuronVar, zero);

            // Constraints that depend on activation
            activeLeq = logic->mkLeq(neuronVar, varInput);
            inactiveLeq = logic->mkLeq(neuronVar, zero);

            addTerm(activeGeq);
            addTerm(inactiveGeq);

            addTerm(logic->mkOr(activeLeq, inactiveLeq));
        } else {
            // Constraints that depend on activation
            PTRef activeEq = activeEqF(*logic, neuronVar, varInput);
            PTRef inactiveEq = inactiveEqF(*logic, neuronVar, zero);

            PTRef activeImpl = logic->mkImpl(activeCond, activeEq);
            PTRef inactiveImpl = logic->mkImpl(inactiveCond, inactiveEq);

            addTerm(activeImpl);
            addTerm(inactiveImpl);
        }
    }

    if (auto optPreferredActivation = verifier.getPreferredNeuronActivation(layer, node)) {
        if (*optPreferredActivation) {
            addPreference(encodeNeuronVars and encodeReluLowerBounds ? activeLeq : activeCond);
        } else {
            addPreference(encodeNeuronVars and encodeReluLowerBounds ? inactiveLeq : inactiveCond);
        }
    }

    if (not storeNeuronTerms) { return PTRef_Undef; }

    return iteF(*logic, activeCond, termInput, zero);
}

void OpenSMTVerifier2::OpenSMTImpl::setUnsatCoreFilter(std::vector<NodeIndex> const & filter) {
    unsatCoreNodeFilter = filter;
}

void OpenSMTVerifier2::OpenSMTImpl::addTerm(PTRef const & term) {
    solver->addAssertion(term);
}

void OpenSMTVerifier2::OpenSMTImpl::addExplanationTerm(PTRef const & term, std::string termNamePrefix) {
    addTerm(term);
    std::size_t const termIdx = explanationTerms.size();
    explanationTerms.push_back(term);

    if (not unsatCoreNodeFilter.empty()) {
        if (std::ranges::none_of(unsatCoreNodeFilter, [this, &term](NodeIndex node) { return contains(term, node); })) {
            unsatCoreFilteredTermsToIndex[term] = termIdx;
            return;
        }
    }
    unsatCoreNonFilteredTermsToIndex[term] = termIdx;

    [[maybe_unused]] bool const success =
        solver->tryAddTermNameFor(term, makeExplanationTermName(std::move(termNamePrefix)));
    assert(success);
}

PTRef OpenSMTVerifier2::OpenSMTImpl::makeUpperBound(LayerIndex layer, NodeIndex node, FastRational value) {
    if (layer != 0 and layer != layerSizes.size() - 1) { throw std::logic_error("Unimplemented!"); }
    PTRef var = layer == 0 ? inputVars.at(node) : outputVarsOrTerms.at(node);
    return logic->mkLeq(var, logic->mkRealConst(value));
}

PTRef OpenSMTVerifier2::OpenSMTImpl::makeLowerBound(LayerIndex layer, NodeIndex node, FastRational value) {
    if (layer != 0 and layer != layerSizes.size() - 1) { throw std::logic_error("Unimplemented!"); }
    PTRef var = layer == 0 ? inputVars.at(node) : outputVarsOrTerms.at(node);
    return logic->mkGeq(var, logic->mkRealConst(value));
}

PTRef OpenSMTVerifier2::OpenSMTImpl::makeInterval(LayerIndex layer, NodeIndex node, FastRational lo, FastRational hi) {
    PTRef lterm = makeLowerBound(layer, node, std::move(lo));
    PTRef uterm = makeUpperBound(layer, node, std::move(hi));
    return logic->mkAnd(lterm, uterm);
}

PTRef OpenSMTVerifier2::OpenSMTImpl::addUpperBound(LayerIndex layer, NodeIndex node, Float value, bool explanationTerm) {
    PTRef term = makeUpperBound(layer, node, value);
    if (not explanationTerm) {
        addTerm(term);
        return term;
    }

    assert(layer == 0);
    addExplanationTerm(term, "u_");
    auto const [_, inserted] = inputVarUpperBoundToIndex.emplace(term, node);
    assert(inserted);

    return term;
}

PTRef OpenSMTVerifier2::OpenSMTImpl::addLowerBound(LayerIndex layer, NodeIndex node, Float value, bool explanationTerm) {
    PTRef term = makeLowerBound(layer, node, value);
    if (not explanationTerm) {
        addTerm(term);
        return term;
    }

    assert(layer == 0);
    addExplanationTerm(term, "l_");
    auto const [_, inserted] = inputVarLowerBoundToIndex.emplace(term, node);
    assert(inserted);

    return term;
}

PTRef OpenSMTVerifier2::OpenSMTImpl::addEquality(LayerIndex layer, NodeIndex node, Float value, bool explanationTerm) {
    PTRef term = makeEquality(layer, node, value);
    if (not explanationTerm) {
        addTerm(term);
        return term;
    }

    assert(layer == 0);
    addExplanationTerm(term, "e_");
    auto const [_, inserted] = inputVarEqualityToIndex.emplace(term, node);
    assert(inserted);

    return term;
}

PTRef OpenSMTVerifier2::OpenSMTImpl::addInterval(LayerIndex layer, NodeIndex node, Float lo, Float hi,
                                                 bool explanationTerm) {
    PTRef term = makeInterval(layer, node, lo, hi);
    if (not explanationTerm) {
        addTerm(term);
        return term;
    }

    assert(layer == 0);
    addExplanationTerm(term, "i_");
    auto const [_, inserted] = inputVarIntervalToIndex.emplace(term, node);
    assert(inserted);

    return term;
}

void OpenSMTVerifier2::OpenSMTImpl::addClassificationConstraint(NodeIndex node, Float threshold = 0.0) {
    // Ensure the node index is within the range of outputVarsOrTerms
    if (node >= outputVarsOrTerms.size()) {
        throw std::out_of_range("Node index is out of range for outputVarsOrTerms.");
    }

    PTRef targetNodeVar = outputVarsOrTerms[node];
    std::vector<PTRef> constraints;

    for (size_t i = 0; i < outputVarsOrTerms.size(); ++i) {
        if (i != node) {
            // Create a constraint: (targetNodeVar - outputVarsOrTerms[i]) > threshold
            PTRef diff = logic->mkMinus(outputVarsOrTerms[i], targetNodeVar);
            PTRef thresholdConst = logic->mkRealConst(floatToRational(threshold));
            PTRef constraint = logic->mkGt(diff, thresholdConst);
            constraints.push_back(constraint);
        }
    }

    if (!constraints.empty()) {
        PTRef combinedConstraint = logic->mkOr(constraints);
        addTerm(combinedConstraint);
    }
}

void OpenSMTVerifier2::OpenSMTImpl::addConstraint(LayerIndex, std::vector<std::pair<NodeIndex, int>>, Float) {
    throw std::logic_error("Unimplemented!");
}

void OpenSMTVerifier2::OpenSMTImpl::addPreference(PTRef const & term) {
    solver->addDecisionPreference(term);
}

void OpenSMTVerifier2::OpenSMTImpl::push() {
    solver->push();
}

void OpenSMTVerifier2::OpenSMTImpl::pop() {
    solver->pop();
}

void OpenSMTVerifier2::OpenSMTImpl::setTimeLimit(std::chrono::milliseconds limit) {
    solver->setTimeLimit(limit);
}

Verifier::Answer OpenSMTVerifier2::OpenSMTImpl::check() {
    auto res = solver->check();
    return toAnswer(res);
}

void OpenSMTVerifier2::OpenSMTImpl::init() {
    config = std::make_unique<SMTConfig>();
    char const * msg = "ok";

    // Must be set before initialization
    config->setProduceProofs();
    config->setOption(SMTConfig::o_produce_inter, SMTOption(true), msg);

    // reset() is called by Verifier
}

void OpenSMTVerifier2::OpenSMTImpl::resetSampleQuery() {
    explanationTerms.clear();
    unsatCoreNonFilteredTermsToIndex.clear();
    unsatCoreFilteredTermsToIndex.clear();

    inputVarLowerBoundToIndex.clear();
    inputVarUpperBoundToIndex.clear();
    inputVarEqualityToIndex.clear();
    inputVarIntervalToIndex.clear();
}

void OpenSMTVerifier2::OpenSMTImpl::resetSample() {
    // resetSampleQuery() is called by Verifier
}

//+ move sth. common into resetGroundModel
void OpenSMTVerifier2::OpenSMTImpl::resetSampleModel() {
    logic = std::make_unique<ArithLogic>(opensmt::Logic_t::QF_LRA);
    solver = std::make_unique<MainSolver>(*logic, *config, "verifier");
    inputVars.clear();
    neuronVars.clear();
    outputVarsOrTerms.clear();
    inputVarsMap.clear();
    neuronVarsMap.clear();
    outputVarsMap.clear();
    neuronTerms.clear();

    layerSizes.clear();
    nextHiddenLayerIndex = 1;
    nextPoolIndex = 0;

    sampleModelRestrictions.clear();

    // resetSample() is called by Verifier
}

std::unique_ptr<spexplain::Explanation>
OpenSMTVerifier2::OpenSMTImpl::getSampleModelRestrictions(spexplain::Framework const & framework) {
    spexplain::ConjunctExplanation cexplanation{framework};
    for (PTRef rest : sampleModelRestrictions) {
        auto phiexplanationPtr = std::make_unique<spexplain::opensmt::FormulaExplanation>(framework, rest);
        cexplanation.insertExplanation(std::move(phiexplanationPtr));
    }

    return MAKE_UNIQUE(std::move(cexplanation));
}

UnsatCore OpenSMTVerifier2::OpenSMTImpl::getUnsatCore() const {
    auto const unsatCore = solver->getUnsatCore();
    auto const & unsatCoreTerms = unsatCore->getTerms();

    assert(unsatCoreTerms.size() > 0 or not unsatCoreNodeFilter.empty());

    std::size_t const termsSize = explanationTerms.size();
    assert(termsSize >= unsatCoreTerms.size());
    assert(termsSize > 0);

    UnsatCore unsatCoreRes;
    auto & [includedIndices, excludedIndices, lowerBounds, upperBounds, equalities, intervals] = unsatCoreRes;
    includedIndices.reserve(termsSize);

    auto const includeTerm = [&](PTRef term, std::size_t termIdx) {
        includedIndices.push_back(termIdx);

        bool const containsLower = containsInputLowerBound(term);
        if (containsLower) {
            lowerBounds.push_back(nodeIndexOfInputLowerBound(term));
            return;
        }

        bool const containsUpper = containsInputUpperBound(term);
        if (containsUpper) {
            upperBounds.push_back(nodeIndexOfInputUpperBound(term));
            return;
        }

        bool const containsEquality = containsInputEquality(term);
        if (containsEquality) {
            equalities.push_back(nodeIndexOfInputEquality(term));
            return;
        }

        bool const containsInterval = containsInputInterval(term);
        if (containsInterval) {
            intervals.push_back(nodeIndexOfInputInterval(term));
            return;
        }

        // formulas not related to particular variables must be handled via (in|ex)cluded indices
    };

    assert(not unsatCoreNodeFilter.empty() or unsatCoreFilteredTermsToIndex.empty());
    for (auto & [term, termIdx] : unsatCoreFilteredTermsToIndex) {
        includeTerm(term, termIdx);
    }

    for (PTRef term : unsatCoreTerms) {
        assert(unsatCoreNonFilteredTermsToIndex.contains(term));
        std::size_t const termIdx = unsatCoreNonFilteredTermsToIndex.at(term);
        includeTerm(term, termIdx);
    }

    assert(not includedIndices.empty());
    std::ranges::sort(includedIndices);

    assert(termsSize >= includedIndices.size());
    excludedIndices.reserve(termsSize - includedIndices.size());
    std::ranges::set_difference(std::views::iota(0UL, termsSize), includedIndices, std::back_inserter(excludedIndices));

    assert(excludedIndices.size() == termsSize - includedIndices.size());
    assert(std::ranges::is_sorted(excludedIndices));

#ifndef NDEBUG
    decltype(includedIndices) xorIndices;
    std::ranges::set_symmetric_difference(includedIndices, excludedIndices, std::back_inserter(xorIndices));
    assert(std::ranges::equal(xorIndices, std::views::iota(0UL, termsSize)));
#endif

    std::ranges::sort(lowerBounds);
    std::ranges::sort(upperBounds);
    std::ranges::sort(equalities);
    std::ranges::sort(intervals);

    return unsatCoreRes;
}

void OpenSMTVerifier2::OpenSMTImpl::printSmtLib2Query(std::ostream & os) const {
    logic->dumpHeaderToFile(os);

    for (PTRef phi : solver->getCurrentAssertionsView()) {
        // necessary for removing auxiliary ITE terms but yields redundant constraints
        // phi = logic->removeAuxVars(phi);
        os << "(assert " << logic->termToSMT2String(phi) << " )\n";
    }

    logic->dumpChecksatToFile(os);
}

} // namespace xai::verifiers
