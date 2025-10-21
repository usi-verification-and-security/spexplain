#include "OpenSMTVerifier.h"
#include "spexplain/network/ConvLayer.h"
#include "spexplain/network/FCLayer.h"
#include "spexplain/network/FlattenLayer.h"

#include <api/MainSolver.h>
#include <common/StringConv.h>
#include <logics/ArithLogic.h>
#include <logics/LogicFactory.h>

#include <algorithm>
#include <ranges>
#include <string>
#include <unordered_map>

namespace xai::verifiers {

using namespace opensmt;

namespace { // Helper methods
FastRational floatToRational(float value);
}

class OpenSMTVerifier::OpenSMTImpl {
public:
    bool contains(PTRef const &, NodeIndex) const;

    std::size_t termSizeOf(PTRef const &) const;

    void loadModel(spexplain::Network const &);
    LayerRefsVariant encodeFCLayer(spexplain::FCLayer const &, LayerRefsVariant const &);
    LayerRefsVariant encodeConvLayer(spexplain::ConvLayer const &, LayerRefsVariant const &);
    LayerRefsVariant encodeFlattenLayer(spexplain::FlattenLayer const & , LayerRefsVariant const &);

    void setUnsatCoreFilter(std::vector<NodeIndex> const &);

    void addTerm(PTRef const &);
    void addExplanationTerm(PTRef const &, std::string termNamePrefix = "");

    PTRef makeUpperBound(LayerIndex layer, NodeIndex node, float value) {
        return makeUpperBound(layer, node, floatToRational(value));
    }
    PTRef makeLowerBound(LayerIndex layer, NodeIndex node, float value) {
        return makeLowerBound(layer, node, floatToRational(value));
    }
    PTRef makeEquality(LayerIndex layer, NodeIndex node, float value) {
        return makeEquality(layer, node, floatToRational(value));
    }
    PTRef makeInterval(LayerIndex layer, NodeIndex node, float lo, float hi) {
        return makeInterval(layer, node, floatToRational(lo), floatToRational(hi));
    }
    PTRef makeUpperBound(LayerIndex layer, NodeIndex node, FastRational value);
    PTRef makeLowerBound(LayerIndex layer, NodeIndex node, FastRational value);
    PTRef makeEquality(LayerIndex layer, NodeIndex node, FastRational value) {
        FastRational valueCp = value;
        return makeInterval(layer, node, std::move(value), std::move(valueCp));
    }
    PTRef makeInterval(LayerIndex layer, NodeIndex node, FastRational lo, FastRational hi);

    PTRef addUpperBound(LayerIndex layer, NodeIndex node, float value, bool explanationTerm = false);
    PTRef addLowerBound(LayerIndex layer, NodeIndex node, float value, bool explanationTerm = false);
    PTRef addEquality(LayerIndex layer, NodeIndex node, float value, bool explanationTerm = false);
    PTRef addInterval(LayerIndex layer, NodeIndex node, float lo, float hi, bool explanationTerm = false);

    void addClassificationConstraint(NodeIndex node, float threshold);

    void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, float rhs);

    void init();

    void push();
    void pop();

    Answer check();

    void resetSampleQuery();
    void resetSample();
    void reset();

    UnsatCore getUnsatCore() const;

    opensmt::MainSolver const & getSolver() const { return *solver; }
    opensmt::MainSolver & getSolver() { return *solver; }

    void printSmtLib2Query(std::ostream &) const;
    LayerRefsVariant reshapeInputVars(std::vector<std::size_t>);

private:
    //! sync with the framework
    static std::string inputVarName(NodeIndex node) {
        return "x" + std::to_string(node + 1);
    }

    std::string makeExplanationTermName(std::string prefix = "") {
        return prefix + "t" + std::to_string(explanationTerms.size() - 1);
    }

    bool containsInputLowerBound(PTRef term) const { return inputVarLowerBoundToIndex.contains(term); }
    bool containsInputUpperBound(PTRef term) const { return inputVarUpperBoundToIndex.contains(term); }
    bool containsInputEquality(PTRef term) const { return inputVarEqualityToIndex.contains(term); }
    bool containsInputInterval(PTRef term) const { return inputVarIntervalToIndex.contains(term); }
    NodeIndex nodeIndexOfInputLowerBound(PTRef term) const { return inputVarLowerBoundToIndex.at(term); }
    NodeIndex nodeIndexOfInputUpperBound(PTRef term) const { return inputVarUpperBoundToIndex.at(term); }
    NodeIndex nodeIndexOfInputEquality(PTRef term) const { return inputVarEqualityToIndex.at(term); }
    NodeIndex nodeIndexOfInputInterval(PTRef term) const { return inputVarIntervalToIndex.at(term); }

    std::unique_ptr<ArithLogic> logic;
    std::unique_ptr<MainSolver> solver;
    std::unique_ptr<SMTConfig> config;
    std::vector<PTRef> inputVars;
    std::vector<PTRef> outputVars;
    std::size_t numLayers;

    std::vector<NodeIndex> unsatCoreNodeFilter;

    std::vector<PTRef> explanationTerms;
    std::unordered_map<PTRef, std::size_t, PTRefHash> unsatCoreNonFilteredTermsToIndex;
    std::unordered_map<PTRef, std::size_t, PTRefHash> unsatCoreFilteredTermsToIndex;

    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarLowerBoundToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarUpperBoundToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarEqualityToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarIntervalToIndex;
};

OpenSMTVerifier::OpenSMTVerifier() : pimpl{std::make_unique<OpenSMTImpl>()} {}

OpenSMTVerifier::~OpenSMTVerifier() {}

bool OpenSMTVerifier::contains(PTRef const & term, NodeIndex node) const {
    return pimpl->contains(term, node);
}

std::size_t OpenSMTVerifier::termSizeOf(PTRef const & term) const {
    return pimpl->termSizeOf(term);
}

void OpenSMTVerifier::loadModel(spexplain::Network const & network) {
    pimpl->loadModel(network);
}

void OpenSMTVerifier::setUnsatCoreFilter(std::vector<NodeIndex> const & filter) {
    pimpl->setUnsatCoreFilter(filter);
}

void OpenSMTVerifier::addTerm(PTRef const & term) {
    pimpl->addTerm(term);
}

void OpenSMTVerifier::addExplanationTerm(PTRef const & term, std::string termNamePrefix) {
    pimpl->addExplanationTerm(term, std::move(termNamePrefix));
}

void OpenSMTVerifier::addUpperBound(LayerIndex layer, NodeIndex var, float value, bool explanationTerm) {
    pimpl->addUpperBound(layer, var, value, explanationTerm);
}

void OpenSMTVerifier::addLowerBound(LayerIndex layer, NodeIndex var, float value, bool explanationTerm) {
    pimpl->addLowerBound(layer, var, value, explanationTerm);
}

void OpenSMTVerifier::addEquality(LayerIndex layer, NodeIndex var, float value, bool explanationTerm) {
    pimpl->addEquality(layer, var, value, explanationTerm);
}

void OpenSMTVerifier::addInterval(LayerIndex layer, NodeIndex var, float lo, float hi, bool explanationTerm) {
    pimpl->addInterval(layer, var, lo, hi, explanationTerm);
}

void OpenSMTVerifier::addClassificationConstraint(NodeIndex node, float threshold=0) {
    pimpl->addClassificationConstraint(node, threshold);
}

void OpenSMTVerifier::addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, float rhs) {
    pimpl->addConstraint(layer, lhs, rhs);
}

void OpenSMTVerifier::initImpl() {
    pimpl->init();
}

void OpenSMTVerifier::pushImpl() {
    pimpl->push();
}

void OpenSMTVerifier::popImpl() {
    pimpl->pop();
}

Verifier::Answer OpenSMTVerifier::checkImpl() {
    return pimpl->check();
}

void OpenSMTVerifier::resetSampleQuery() {
    pimpl->resetSampleQuery();
    UnsatCoreVerifier::resetSampleQuery();
}

void OpenSMTVerifier::resetSample() {
    pimpl->resetSample();
    UnsatCoreVerifier::resetSample();
}

void OpenSMTVerifier::reset() {
    pimpl->reset();
    UnsatCoreVerifier::reset();
}

UnsatCore OpenSMTVerifier::getUnsatCore() const {
    return pimpl->getUnsatCore();
}

opensmt::MainSolver const & OpenSMTVerifier::getSolver() const {
    return pimpl->getSolver();
}

opensmt::MainSolver & OpenSMTVerifier::getSolver() {
    return pimpl->getSolver();
}

void OpenSMTVerifier::printSmtLib2Query(std::ostream & os) const {
    return pimpl->printSmtLib2Query(os);
}

/*
 * Actual implementation
 */

namespace { // Helper methods
FastRational floatToRational(float value) {
    auto s = std::to_string(value);
    char* rationalString;
    opensmt::stringToRational(rationalString, s.c_str());
    auto res = FastRational(rationalString);
    free(rationalString);
    return res;
}

Verifier::Answer toAnswer(sstat res) {
    if (res == s_False)
        return Verifier::Answer::UNSAT;
    if (res == s_True)
        return Verifier::Answer::SAT;
    if (res == s_Error)
        return Verifier::Answer::ERROR;
    if (res == s_Undef)
        return Verifier::Answer::UNKNOWN;
    return Verifier::Answer::UNKNOWN;
}
}

bool OpenSMTVerifier::OpenSMTImpl::contains(PTRef const & term, NodeIndex node) const {
    auto & solver = getSolver();
    auto & logic = solver.getLogic();

    auto & inputVar = inputVars.at(node);
    return logic.contains(term, inputVar);
}

std::size_t OpenSMTVerifier::OpenSMTImpl::termSizeOf(PTRef const & term) const {
    auto & solver = getSolver();
    auto & logic = solver.getLogic();
    Pterm const & pterm = logic.getPterm(term);

    if (logic.isAtom(term)) { return 1; }

    if (logic.isNot(term)) {
        assert(pterm.size() == 1);
        auto & negTerm = *pterm.begin();
        assert(not logic.isNot(negTerm));
        // just care about no. literals, so ignore the negations themselves
        return termSizeOf(negTerm);
    }

    assert(logic.isAnd(term) or logic.isOr(term));
    std::size_t totalSize{};
    for (PTRef const & argTerm : pterm) {
        totalSize += termSizeOf(argTerm);
    }
    return totalSize;
}

void OpenSMTVerifier::OpenSMTImpl::loadModel(spexplain::Network const & network) {
    numLayers = network.getNumLayers();
    // create input variables
    auto numInputs = network.getInputSizeFlat();
    for (NodeIndex i = 0u; i < numInputs; ++i) {
        auto name = inputVarName(i);
        PTRef var = logic->mkRealVar(name.c_str());
        inputVars.push_back(var);
    }

    // Create representation for each neuron in hidden layers, from input to output layers
    LayerRefsVariant previousLayerRefs = reshapeInputVars(network.getInputSize());
    for (LayerIndex layerInd = 0u; layerInd < network.getNumLayers(); layerInd++) {
        LayerRefsVariant currentLayerRefs;
        auto layer = network.getLayer(layerInd);
        if (auto fcLayer = dynamic_cast<spexplain::FCLayer const *>(layer)) {
            currentLayerRefs = std::get<PTRef1D>(encodeFCLayer(*fcLayer, previousLayerRefs));
        } else if (auto convLayer = dynamic_cast<spexplain::ConvLayer const *>(layer)) {
            currentLayerRefs = std::get<PTRef3D>(encodeConvLayer(*convLayer, previousLayerRefs));
        } else if (auto flattenLayer = dynamic_cast<spexplain::FlattenLayer const *>(layer)) {
            currentLayerRefs = std::get<PTRef1D>(encodeFlattenLayer(*flattenLayer, previousLayerRefs));
        } else {
            throw std::logic_error("Unsupported layer type!");
        }
        previousLayerRefs = std::move(currentLayerRefs);
    }
    outputVars.clear();

    assert(std::holds_alternative<PTRef1D>(previousLayerRefs));
    const auto &prevLayerRefsVec = std::get<PTRef1D>(previousLayerRefs);
    outputVars = std::move(const_cast<PTRef1D&>(prevLayerRefsVec));

    // Collect hard bounds on inputs
    std::vector<PTRef> bounds;
    assert(network.getInputSizeFlat() == inputVars.size());
    for (NodeIndex i = 0; i < inputVars.size(); ++i) {
        float lb = network.getInputLowerBound(i);
        float ub = network.getInputUpperBound(i);
        bounds.push_back(logic->mkGeq(inputVars[i], logic->mkRealConst(floatToRational(lb))));
        bounds.push_back(logic->mkLeq(inputVars[i], logic->mkRealConst(floatToRational(ub))));
    }
    addTerm(logic->mkAnd(bounds));
    std::cout <<  "Encoding the network is done successfully!" << std::endl;
}

OpenSMTVerifier::LayerRefsVariant OpenSMTVerifier::OpenSMTImpl::encodeFCLayer(spexplain::FCLayer const & layer, LayerRefsVariant const & previousLayerRefs) {
    assert(std::holds_alternative<PTRef1D>(previousLayerRefs));
    PTRef1D currentLayerRefs;
    assert(layer.getLayerSize().size() == 1); // only 1D supported for FC
    auto const biasVec = layer.getBias();

    for (NodeIndex node = 0u; node < layer.getLayerSize()[0]; ++node) {
        PTRef1D addends;
        float bias = biasVec[node];
        auto const weights = std::get<spexplain::Network::Values>(layer.getWeights(node));
        PTRef biasTerm = logic->mkRealConst(floatToRational(bias));
        addends.push_back(biasTerm);
        auto const & prevLayerRefsVec = std::get<PTRef1D>(previousLayerRefs);
        assert(prevLayerRefsVec.size() == weights.size());
        for (int j = 0; j < weights.size(); j++) {
        PTRef weightTerm = logic->mkRealConst(floatToRational(weights[j]));
        PTRef addend = logic->mkTimes(weightTerm, prevLayerRefsVec[j]);
        addends.push_back(addend);
        }
        PTRef input = logic->mkPlus(addends);
        if (layer.followsByRelu()) {
            // Apply ReLU
            PTRef relu = logic->mkIte(logic->mkGeq(input, logic->getTerm_RealZero()), input, logic->getTerm_RealZero());
            currentLayerRefs.push_back(relu);
        }else {
            currentLayerRefs.push_back(input);
        }
    }
    return currentLayerRefs;
}

OpenSMTVerifier::LayerRefsVariant OpenSMTVerifier::OpenSMTImpl::encodeConvLayer(spexplain::ConvLayer const & layer, LayerRefsVariant const & previousLayerRefs) {
    assert(std::holds_alternative<PTRef3D>(previousLayerRefs) or
           std::holds_alternative<PTRef2D>(previousLayerRefs));  //2D or 3D inputs
    // If 2D inputs, convert to 3D
    PTRef3D previousLayerRefs3D;
    if (std::holds_alternative<PTRef2D>(previousLayerRefs)) {
        auto const & prevLayerRefs2D = std::get<PTRef2D>(previousLayerRefs);
        previousLayerRefs3D.push_back(prevLayerRefs2D);
    } else {
        auto &previousLayerRefs3D = std::get<PTRef3D>(previousLayerRefs);
    }
    PTRef3D currentLayerRefs;
    assert(layer.getLayerSize().size() == 3); // only 3D supported for Conv
    auto const filters = std::get<spexplain::NetworkLayer::Vector4D>(layer.getWeights());
    auto const biasVec = layer.getBias();
    int stride = layer.getStride();
    int padding = layer.getPadding();

    std::size_t inChannels = previousLayerRefs3D.size();
    std::size_t inRows = previousLayerRefs3D[0].size();
    std::size_t inCols = previousLayerRefs3D[0][0].size();

    // Pad input: create a new 3D vector of PTRef with zeros for padding
    PTRef3D inputPadded;
    for (std::size_t ic = 0; ic < inChannels; ++ic) {
        PTRef2D paddedChannel(inRows + 2 * padding, PTRef1D(inCols + 2 * padding, logic->getTerm_RealZero()));
        for (std::size_t i = 0; i < inRows; ++i)
            for (std::size_t j = 0; j < inCols; ++j)
                paddedChannel[i + padding][j + padding] = previousLayerRefs3D[ic][i][j];
        inputPadded.push_back(std::move(paddedChannel));
    }

    inChannels = inputPadded.size();
    inRows = inputPadded[0].size();
    inCols = inputPadded[0][0].size();

    std::size_t outChannels = filters.size();
    std::size_t filterRows = filters[0][0].size();
    std::size_t filterCols = filters[0][0][0].size();

    std::size_t outRows = (inRows - filterRows) / layer.getStride() + 1;
    std::size_t outCols = (inCols - filterCols) / layer.getStride() + 1;

    assert(layer.getLayerSize()[0] == filters.size());

    // Convolution encoding
    for (std::size_t oc = 0; oc < outChannels; ++oc) {
        PTRef2D outputChannel;
        for (std::size_t i = 0; i < outRows; ++i) {
            PTRef1D outputRow;
            for (std::size_t j = 0; j < outCols; ++j) {
                PTRef1D addends;
                // Add bias
                float bias = biasVec[oc];
                addends.push_back(logic->mkRealConst(floatToRational(bias)));
                // Convolution sum
                for (std::size_t ic = 0; ic < inChannels; ++ic) {
                    for (std::size_t m = 0; m < filterRows; ++m) {
                        for (std::size_t n = 0; n < filterCols; ++n) {
                            std::size_t x = i * stride + m;
                            std::size_t y = j * stride + n;
                            PTRef inputTerm = inputPadded[ic][x][y];
                            float weight = filters[oc][ic][m][n];
                            PTRef weightTerm = logic->mkRealConst(floatToRational(weight));
                            addends.push_back(logic->mkTimes(weightTerm, inputTerm));
                        }
                    }
                }
                PTRef sum = logic->mkPlus(addends);
                // Apply ReLU
                if (layer.followsByRelu()) {
                    PTRef relu = logic->mkIte(logic->mkGeq(sum, logic->getTerm_RealZero()), sum, logic->getTerm_RealZero());
                    outputRow.push_back(relu);
                }else {
                    outputRow.push_back(sum);
                }
            }
            outputChannel.push_back(std::move(outputRow));
        }
        currentLayerRefs.push_back(std::move(outputChannel));
    }
    return currentLayerRefs;
}

OpenSMTVerifier::LayerRefsVariant OpenSMTVerifier::OpenSMTImpl::encodeFlattenLayer(spexplain::FlattenLayer const & layer, LayerRefsVariant const & previousLayerRefs) {
    assert(std::holds_alternative<PTRef2D>(previousLayerRefs) or std::holds_alternative<PTRef3D>(previousLayerRefs));
    PTRef1D currentLayerRefs;
    if (std::holds_alternative<PTRef2D>(previousLayerRefs)) {
        auto const & prevLayerRefs2D = std::get<PTRef2D>(previousLayerRefs);
        for (auto const & row : prevLayerRefs2D) {
            for (auto const & elem : row) {
                currentLayerRefs.push_back(elem);
            }
        }
        return currentLayerRefs;
    } else {
        auto const & prevLayerRefs3D = std::get<PTRef3D>(previousLayerRefs);
        for (auto const & matrix : prevLayerRefs3D) {
            for (auto const & row : matrix) {
                for (auto const & elem : row) {
                    currentLayerRefs.push_back(elem);
                }
            }
        }
        return currentLayerRefs;
    }
}

void OpenSMTVerifier::OpenSMTImpl::setUnsatCoreFilter(std::vector<NodeIndex> const & filter) {
    unsatCoreNodeFilter = filter;
}

void OpenSMTVerifier::OpenSMTImpl::addTerm(PTRef const & term) {
    solver->addAssertion(term);
}

void OpenSMTVerifier::OpenSMTImpl::addExplanationTerm(PTRef const & term, std::string termNamePrefix) {
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

    [[maybe_unused]] bool const success = solver->tryAddTermNameFor(term, makeExplanationTermName(std::move(termNamePrefix)));
    assert(success);
}

PTRef OpenSMTVerifier::OpenSMTImpl::makeUpperBound(LayerIndex layer, NodeIndex node, FastRational value) {
    if (layer != 0 and layer != numLayers - 1)
        throw std::logic_error("Unimplemented!");
    PTRef var = layer == 0 ? inputVars.at(node) : outputVars.at(node);
    return logic->mkLeq(var, logic->mkRealConst(value));
}

PTRef OpenSMTVerifier::OpenSMTImpl::makeLowerBound(LayerIndex layer, NodeIndex node, FastRational value) {
    if (layer != 0 and layer != numLayers - 1)
        throw std::logic_error("Unimplemented!");
    PTRef var = layer == 0 ? inputVars.at(node) : outputVars.at(node);
    return logic->mkGeq(var, logic->mkRealConst(value));
}

PTRef OpenSMTVerifier::OpenSMTImpl::makeInterval(LayerIndex layer, NodeIndex node, FastRational lo, FastRational hi) {
    PTRef lterm = makeLowerBound(layer, node, std::move(lo));
    PTRef uterm = makeUpperBound(layer, node, std::move(hi));
    return logic->mkAnd(lterm, uterm);
}

PTRef OpenSMTVerifier::OpenSMTImpl::addUpperBound(LayerIndex layer, NodeIndex node, float value, bool explanationTerm) {
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

PTRef OpenSMTVerifier::OpenSMTImpl::addLowerBound(LayerIndex layer, NodeIndex node, float value, bool explanationTerm) {
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

PTRef OpenSMTVerifier::OpenSMTImpl::addEquality(LayerIndex layer, NodeIndex node, float value, bool explanationTerm) {
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

PTRef OpenSMTVerifier::OpenSMTImpl::addInterval(LayerIndex layer, NodeIndex node, float lo, float hi, bool explanationTerm) {
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

void OpenSMTVerifier::OpenSMTImpl::addClassificationConstraint(NodeIndex node, float threshold=0.0){
    // Ensure the node index is within the range of outputVars
    if (node >= outputVars.size()) {
        throw std::out_of_range("Node index is out of range for outputVars.");
    }

    PTRef targetNodeVar = outputVars[node];
    std::vector<PTRef> constraints;

    for (size_t i = 0; i < outputVars.size(); ++i) {
        if (i != node) {
            // Create a constraint: (targetNodeVar - outputVars[i]) > threshold
            PTRef diff = logic->mkMinus(outputVars[i], targetNodeVar);
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

void
OpenSMTVerifier::OpenSMTImpl::addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, float rhs) {
    throw std::logic_error("Unimplemented!");
}

void OpenSMTVerifier::OpenSMTImpl::push() {
    solver->push();
}

void OpenSMTVerifier::OpenSMTImpl::pop() {
    solver->pop();
}

Verifier::Answer OpenSMTVerifier::OpenSMTImpl::check() {
    auto res = solver->check();
    return toAnswer(res);
}

void OpenSMTVerifier::OpenSMTImpl::init() {
    config = std::make_unique<SMTConfig>();
    char const * msg = "ok";

    // Must be set before initialization
    config->setProduceProofs();
    config->setOption(SMTConfig::o_produce_inter, SMTOption(true), msg);

    // reset() is called by Verifier
}

void OpenSMTVerifier::OpenSMTImpl::resetSampleQuery() {
    explanationTerms.clear();
    unsatCoreNonFilteredTermsToIndex.clear();
    unsatCoreFilteredTermsToIndex.clear();

    inputVarLowerBoundToIndex.clear();
    inputVarUpperBoundToIndex.clear();
    inputVarEqualityToIndex.clear();
    inputVarIntervalToIndex.clear();
}

void OpenSMTVerifier::OpenSMTImpl::resetSample() {
    // resetSampleQuery() is called by Verifier
}

void OpenSMTVerifier::OpenSMTImpl::reset() {
    logic = std::make_unique<ArithLogic>(opensmt::Logic_t::QF_LRA);
    solver = std::make_unique<MainSolver>(*logic, *config, "verifier");
    inputVars.clear();
    outputVars.clear();

    // resetSample() is called by Verifier
}

UnsatCore OpenSMTVerifier::OpenSMTImpl::getUnsatCore() const {
    auto const unsatCore = solver->getUnsatCore();
    auto const & unsatCoreTerms = unsatCore->getTerms();

    assert(unsatCoreTerms.size() > 0 or not unsatCoreNodeFilter.empty());

    std::size_t const termsSize = explanationTerms.size();
    assert(termsSize >= unsatCoreTerms.size());
    assert(termsSize > 0);

    UnsatCore unsatCoreRes;
    auto & [includedIndices, excludedIndices, lowerBounds, upperBounds, equalities, intervals] = unsatCoreRes;
    includedIndices.reserve(termsSize);

    auto const includeTerm = [&](PTRef term, std::size_t termIdx){
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

void OpenSMTVerifier::OpenSMTImpl::printSmtLib2Query(std::ostream & os) const {
    auto & solver = getSolver();
    auto & logic = solver.getLogic();
    logic.dumpHeaderToFile(os);

    for (PTRef phi : solver.getCurrentAssertionsView()) {
        // necessary for removing auxiliary ITE terms but yields redundant constraints
        // phi = logic.removeAuxVars(phi);
        os << "(assert " << logic.printTerm(phi) << " )\n";
    }

    logic.dumpChecksatToFile(os);
}

OpenSMTVerifier::LayerRefsVariant OpenSMTVerifier::OpenSMTImpl::reshapeInputVars(std::vector<std::size_t> shape) {
    if (shape.size() == 1) {
        PTRef1D inputVars1D;
        for (auto const & var : inputVars) {
            inputVars1D.push_back(var);
        }
        return inputVars1D;
    } else if (shape.size() == 2) {
        PTRef2D inputVars2D;
        std::size_t rows = shape[0];
        std::size_t cols = shape[1];
        assert(rows * cols == inputVars.size());
        for (std::size_t i = 0; i < rows; ++i) {
            PTRef1D row;
            for (std::size_t j = 0; j < cols; ++j) {
                row.push_back(inputVars[i * cols + j]);
            }
            inputVars2D.push_back(row);
        }
        return inputVars2D;
    } else if (shape.size() == 3) {
        PTRef3D inputVars3D;
        std::size_t depth = shape[0];
        std::size_t rows = shape[1];
        std::size_t cols = shape[2];
        assert(depth * rows * cols == inputVars.size());
        for (std::size_t d = 0; d < depth; ++d) {
            PTRef2D matrix;
            for (std::size_t i = 0; i < rows; ++i) {
                PTRef1D row;
                for (std::size_t j = 0; j < cols; ++j) {
                    row.push_back(inputVars[d * rows * cols + i * cols + j]);
                }
                matrix.push_back(row);
            }
            inputVars3D.push_back(matrix);
        }
        return inputVars3D;
    } else {
        throw std::logic_error("Unsupported input shape!");
    }
}

} // namespace xai::verifiers
