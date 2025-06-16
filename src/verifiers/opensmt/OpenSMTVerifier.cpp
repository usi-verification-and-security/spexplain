#include "OpenSMTVerifier.h"

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
    void loadModel(spexplain::Network const &);

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

private:
    static std::string makeTermName(LayerIndex layer, NodeIndex node, std::string prefix = "") {
        assert(layer == 0);
        return prefix + "n" + std::to_string(node);
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
    std::vector<std::size_t> layerSizes;

    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarLowerBoundToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarUpperBoundToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarEqualityToIndex;
    std::unordered_map<PTRef, NodeIndex, PTRefHash> inputVarIntervalToIndex;
};

OpenSMTVerifier::OpenSMTVerifier() : pimpl{std::make_unique<OpenSMTImpl>()} {}

OpenSMTVerifier::~OpenSMTVerifier() {}

void OpenSMTVerifier::loadModel(spexplain::Network const & network) {
    pimpl->loadModel(network);
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

void OpenSMTVerifier::OpenSMTImpl::loadModel(spexplain::Network const & network) {
    // create input variables
    for (NodeIndex i = 0u; i < network.getLayerSize(0); ++i) {
        auto name = "x" + std::to_string(i + 1);
        PTRef var = logic->mkRealVar(name.c_str());
        inputVars.push_back(var);
    }

    // Create representation for each neuron in hidden layers, from input to output layers
    std::vector<PTRef> previousLayerRefs = inputVars;
    for (LayerIndex layer = 1u; layer < network.getNumLayers() - 1; layer++) {
        std::vector<PTRef> currentLayerRefs;
        for (NodeIndex node = 0u; node < network.getLayerSize(layer); ++node) {
            std::vector<PTRef> addends;
            float bias = network.getBias(layer, node);
            auto const & weights = network.getWeights(layer, node);
            PTRef biasTerm = logic->mkRealConst(floatToRational(bias));
            addends.push_back(biasTerm);

            assert(previousLayerRefs.size() == weights.size());
            for (int j = 0; j < weights.size(); j++) {
                PTRef weightTerm = logic->mkRealConst(floatToRational(weights[j]));
                PTRef addend = logic->mkTimes(weightTerm, previousLayerRefs[j]);
                addends.push_back(addend);
            }
            PTRef input = logic->mkPlus(addends);
            PTRef relu = logic->mkIte(logic->mkGeq(input, logic->getTerm_RealZero()), input, logic->getTerm_RealZero());
            currentLayerRefs.push_back(relu);
        }
        previousLayerRefs = std::move(currentLayerRefs);
    }

    // Create representation of the outputs (without RELU!)
    // TODO: Remove code duplication!
    auto lastLayerIndex = network.getNumLayers() - 1;
    auto lastLayerSize = network.getLayerSize(lastLayerIndex);
    outputVars.clear();
    for (NodeIndex node = 0u; node < lastLayerSize; ++node) {
        std::vector<PTRef> addends;
        float bias = network.getBias(lastLayerIndex, node);
        auto const & weights = network.getWeights(lastLayerIndex, node);
        PTRef biasTerm = logic->mkRealConst(floatToRational(bias));
        addends.push_back(biasTerm);

        assert(previousLayerRefs.size() == weights.size());
        for (int j = 0; j < weights.size(); j++) {
            PTRef weightTerm = logic->mkRealConst(floatToRational(weights[j]));
            PTRef addend = logic->mkTimes(weightTerm, previousLayerRefs[j]);
            addends.push_back(addend);
        }
        outputVars.push_back(logic->mkPlus(addends));
    }

    // Store information about layer sizes
    layerSizes.clear();
    for (LayerIndex layer = 0u; layer < network.getNumLayers(); layer++) {
        layerSizes.push_back(network.getLayerSize(layer));
    }

    // Collect hard bounds on inputs
    std::vector<PTRef> bounds;
    for (NodeIndex i = 0; i < network.getLayerSize(0); ++i) {
        float lb = network.getInputLowerBound(i);
        float ub = network.getInputUpperBound(i);
        bounds.push_back(logic->mkGeq(inputVars[i], logic->mkRealConst(floatToRational(lb))));
        bounds.push_back(logic->mkLeq(inputVars[i], logic->mkRealConst(floatToRational(ub))));
    }
    solver->addAssertion(logic->mkAnd(bounds));
}

PTRef OpenSMTVerifier::OpenSMTImpl::makeUpperBound(LayerIndex layer, NodeIndex node, FastRational value) {
    if (layer != 0 and layer != layerSizes.size() - 1)
        throw std::logic_error("Unimplemented!");
    PTRef var = layer == 0 ? inputVars.at(node) : outputVars.at(node);
    return logic->mkLeq(var, logic->mkRealConst(value));
}

PTRef OpenSMTVerifier::OpenSMTImpl::makeLowerBound(LayerIndex layer, NodeIndex node, FastRational value) {
    if (layer != 0 and layer != layerSizes.size() - 1)
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
    solver->addAssertion(term);
    if (not explanationTerm) { return term; }

    [[maybe_unused]] bool const success = solver->tryAddTermNameFor(term, makeTermName(layer, node, "u_"));
    assert(success);
    auto const [_, inserted] = inputVarUpperBoundToIndex.emplace(term, node);
    assert(inserted);
    return term;
}

PTRef OpenSMTVerifier::OpenSMTImpl::addLowerBound(LayerIndex layer, NodeIndex node, float value, bool explanationTerm) {
    PTRef term = makeLowerBound(layer, node, value);
    solver->addAssertion(term);
    if (not explanationTerm) { return term; }

    [[maybe_unused]] bool const success = solver->tryAddTermNameFor(term, makeTermName(layer, node, "l_"));
    assert(success);
    auto const [_, inserted] = inputVarLowerBoundToIndex.emplace(term, node);
    assert(inserted);
    return term;
}

PTRef OpenSMTVerifier::OpenSMTImpl::addEquality(LayerIndex layer, NodeIndex node, float value, bool explanationTerm) {
    PTRef term = makeEquality(layer, node, value);
    solver->addAssertion(term);
    if (not explanationTerm) { return term; }

    [[maybe_unused]] bool const success = solver->tryAddTermNameFor(term, makeTermName(layer, node, "e_"));
    assert(success);
    auto const [_, inserted] = inputVarEqualityToIndex.emplace(term, node);
    assert(inserted);
    return term;
}

PTRef OpenSMTVerifier::OpenSMTImpl::addInterval(LayerIndex layer, NodeIndex node, float lo, float hi, bool explanationTerm) {
    PTRef term = makeInterval(layer, node, lo, hi);
    solver->addAssertion(term);
    if (not explanationTerm) { return term; }

    [[maybe_unused]] bool const success = solver->tryAddTermNameFor(term, makeTermName(layer, node, "i_"));
    assert(success);
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
        solver->addAssertion(combinedConstraint);
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
    assert(unsatCore->getTerms().size() > 0);

    // This assumes that all explanation terms are the only included terms at the last assertion level
    // Hence interleaving the insertions with non-explanation terms would break this
    auto const & assertions = solver->getAssertionsAtCurrentLevel();
    std::size_t const assertionsSize = assertions.size();
    assert(assertionsSize >= unsatCore->getTerms().size());

    std::unordered_map<PTRef, std::size_t, PTRefHash> termToAssertionIdxMap;
    termToAssertionIdxMap.reserve(assertionsSize);
    for (std::size_t assertionIdx = 0; assertionIdx < assertionsSize; ++assertionIdx) {
        PTRef term = assertions[assertionIdx];
        [[maybe_unused]] auto const [_, inserted] = termToAssertionIdxMap.emplace(term, assertionIdx);
        assert(inserted);
    }
    assert(termToAssertionIdxMap.size() == assertionsSize);

    UnsatCore unsatCoreRes;
    auto & [includedIndices, excludedIndices, lowerBounds, upperBounds, equalities, intervals] = unsatCoreRes;
    includedIndices.reserve(assertionsSize);

    for (PTRef term : unsatCore->getTerms()) {
        std::size_t const assertionIdx = termToAssertionIdxMap[term];
        includedIndices.push_back(assertionIdx);

        bool const containsLower = containsInputLowerBound(term);
        if (containsLower) {
            lowerBounds.push_back(nodeIndexOfInputLowerBound(term));
            continue;
        }

        bool const containsUpper = containsInputUpperBound(term);
        if (containsUpper) {
            upperBounds.push_back(nodeIndexOfInputUpperBound(term));
            continue;
        }

        bool const containsEquality = containsInputEquality(term);
        if (containsEquality) {
            equalities.push_back(nodeIndexOfInputEquality(term));
            continue;
        }

        bool const containsInterval = containsInputInterval(term);
        if (containsInterval) {
            intervals.push_back(nodeIndexOfInputInterval(term));
            continue;
        }

        // formulas not related to particular variables must be handled via (in|ex)cluded indices
    }

    assert(not includedIndices.empty());
    std::ranges::sort(includedIndices);

    assert(assertionsSize >= includedIndices.size());
    excludedIndices.reserve(assertionsSize - includedIndices.size());
    std::ranges::set_difference(std::views::iota(0UL, assertionsSize), includedIndices, std::back_inserter(excludedIndices));

    assert(excludedIndices.size() == assertionsSize - includedIndices.size());
    assert(std::ranges::is_sorted(excludedIndices));

#ifndef NDEBUG
    decltype(includedIndices) xorIndices;
    std::ranges::set_symmetric_difference(includedIndices, excludedIndices, std::back_inserter(xorIndices));
    assert(std::ranges::equal(xorIndices, std::views::iota(0UL, assertionsSize)));
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

} // namespace xai::verifiers
