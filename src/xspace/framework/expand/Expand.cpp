#include "Expand.h"

#include "../Config.h"
#include "../Preprocess.h"
#include "../Print.h"
#include "../explanation/Explanation.h"
#include "strategy/Factory.h"
#include "strategy/Strategy.h"

#include <xspace/common/Core.h>
#include <xspace/common/String.h>

#include <verifiers/Verifier.h>
#include <verifiers/opensmt/OpenSMTVerifier.h>
#ifdef MARABOU
#include <verifiers/marabou/MarabouVerifier.h>
#endif

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

namespace xspace {
Framework::Expand::Expand(Framework & fw) : framework{fw} {}

void Framework::Expand::setStrategies(std::istream & is) {
    // pipe character '|' reserved for disjunctions
    static constexpr char strategyDelim = ';';

    auto const & config = framework.getConfig();

    VarOrdering defaultVarOrder{};
    if (config.isReverseVarOrdering()) { defaultVarOrder.type = VarOrdering::Type::reverse; }

    Strategy::Factory factory{*this, defaultVarOrder};

    std::string line;
    while (std::getline(is, line, strategyDelim)) {
        auto strategyPtr = factory.parse(line);
        addStrategy(std::move(strategyPtr));
    }
}

void Framework::Expand::addStrategy(std::unique_ptr<Strategy> strategy) {
    requiresSMTSolver |= strategy->requiresSMTSolver();

    strategies.push_back(std::move(strategy));
}

std::unique_ptr<xai::verifiers::Verifier> Framework::Expand::makeVerifier(std::string_view name) const {
#ifdef MARABOU
    if (name.empty() and not requiresSMTSolver) { name = "marabou"sv; }
#endif

    if (name.empty() or toLower(name) == "opensmt") {
        return std::make_unique<xai::verifiers::OpenSMTVerifier>();
#ifdef MARABOU
    } else if (toLower(name) == "marabou") {
        return std::make_unique<xai::verifiers::MarabouVerifier>();
#endif
    }

    throw std::invalid_argument{"Unrecognized verifier name: "s + std::string{name}};
}

void Framework::Expand::setVerifier() {
    setVerifier(""sv);
}

void Framework::Expand::setVerifier(std::string_view name) {
    setVerifier(makeVerifier(name));
}

void Framework::Expand::setVerifier(std::unique_ptr<xai::verifiers::Verifier> vf) {
    assert(vf);
    verifierPtr = std::move(vf);
}

Network::Dataset::SampleIndices Framework::Expand::makeSampleIndices(Network::Dataset const & data) const {
    auto indices = getSampleIndices(data);
    assert(indices.size() <= data.size());

    auto const & config = framework.getConfig();

    if (config.shufflingSamples()) { std::ranges::shuffle(indices, std::default_random_engine{}); }

    if (config.limitingMaxSamples()) {
        auto const maxSamples = config.getMaxSamples();
        assert(maxSamples > 0);
        if (maxSamples < indices.size()) { indices.resize(maxSamples); }
    }

    return indices;
}

Network::Dataset::SampleIndices Framework::Expand::getSampleIndices(Network::Dataset const & data) const {
    auto const & config = framework.getConfig();

    if (not config.filteringSamplesOfExpectedClass()) {
        if (config.filteringCorrectSamples()) { return data.getCorrectSampleIndices(); }
        if (config.filteringIncorrectSamples()) { return data.getIncorrectSampleIndices(); }
        return data.getSampleIndices();
    }

    auto const & label = config.getSamplesExpectedClassFilter();
    if (config.filteringCorrectSamples()) { return data.getCorrectSampleIndicesOfExpectedClass(label); }
    if (config.filteringIncorrectSamples()) { return data.getIncorrectSampleIndicesOfExpectedClass(label); }
    return data.getSampleIndicesOfExpectedClass(label);
}

void Framework::Expand::dumpClassificationsAsSmtLib2Queries() {
    auto & print = framework.getPrint();
    auto & cinfo = print.info();

    auto & network = framework.getNetwork();
    std::size_t nOutputs = network.getOutputSize();
    if (network.isBinaryClassifier()) {
        ++nOutputs;
        assert(nOutputs == 2);
    }

    for (Network::Classification::Label l = 0; l < nOutputs; ++l) {
        Network::Classification cls{.label = l};

        std::string fname = "psi_c" + std::to_string(l) + ".smt2";
        std::ofstream ofs{fname};

        printClassificationAsSmtLib2Query(ofs, cls);

        cinfo << "Dumped classification " << l << " SMT query to: " << fname << '\n';
    }
}

void Framework::Expand::printClassificationAsSmtLib2Query(std::ostream & os, Network::Classification const & cls) {
    auto & verifier = getVerifier();

    auto & label = cls.label;

    initVerifier();

    os << "(set-info :source \"class " << label << "\")\n";
    assertModel();
    assertClassification(cls);

    verifier.printSmtLib2Query(os);
}

void Framework::Expand::operator()(Explanations & explanations, Network::Dataset const & data) {
    assert(not strategies.empty());

    assert(explanations.size() <= data.size());
    //+ if we start from file where filtering (and possibly also shuffling) happened,
    // we would have to sync the explanations with the sample points in the dataset
    // The output variable must also be compatible with this, now we always keep all
    if (explanations.size() < data.size()) {
        throw std::invalid_argument{"Expansion of lower no. explanations than the dataset size is not supported: "s +
                                    std::to_string(explanations.size()) + " < " + std::to_string(data.size())};
    }

    auto & print = framework.getPrint();
    bool const printingStats = not print.ignoringStats();
    bool const printingExplanations = not print.ignoringExplanations();
    auto & cexp = print.explanations();

    if (printingStats) { printStatsHead(data); }

    initVerifier();

    // Such incrementality does not seem to be beneficial
    // assertModel();

    Network::Dataset::SampleIndices const indices = makeSampleIndices(data);
    for (auto idx : indices) {
        // Seems quite more efficient than if outside the loop, at least with 'abductive'
        assertModel();

        auto const & output = data.getComputedOutput(idx);
        auto const & cls = output.classification;
        assertClassification(cls);

        for (auto & strategy : strategies) {
            strategy->execute(explanations, data, idx);
        }

        auto & explanation = getExplanation(explanations, idx);
        //+ get rid of the conditionals
        if (printingStats) { printStats(explanation, data, idx); }
        if (printingExplanations) {
            explanation.print(cexp);
            cexp << std::endl;
        }

        resetClassification();

        resetModel();
    }
}

void Framework::Expand::initVerifier() {
    assert(verifierPtr);
    verifierPtr->init();
}

void Framework::Expand::assertModel() {
    auto & network = framework.getNetwork();
    verifierPtr->loadModel(network);
}

void Framework::Expand::resetModel() {
    verifierPtr->reset();
}

void Framework::Expand::assertClassification(Network::Classification const & cls) {
    verifierPtr->push();

    auto & network = framework.getNetwork();
    auto const outputLayerIndex = network.getNumLayers() - 1;

    auto const & label = cls.label;

    if (not network.isBinaryClassifier()) {
        verifierPtr->addClassificationConstraint(label, 0);
        return;
    }

    // With single output value, the flip in classification means flipping the value across 0
    constexpr Float threshold = 0.015625f;
    assert(label == 0 || label == 1);
    if (label == 1) {
        // <= -threshold
        verifierPtr->addUpperBound(outputLayerIndex, 0, -threshold);
    } else {
        // >= threshold
        verifierPtr->addLowerBound(outputLayerIndex, 0, threshold);
    }
}

void Framework::Expand::resetClassification() {
    verifierPtr->pop();
    verifierPtr->resetSample();
}

void Framework::Expand::printStatsHead(Network::Dataset const & data) const {
    auto & print = framework.getPrint();
    assert(not print.ignoringStats());
    auto & cstats = print.stats();

    auto const & config = framework.getConfig();

    std::size_t const size = data.size();
    cstats << "Dataset size: " << size << '\n';
    if (config.limitingMaxSamples()) {
        auto const maxSamples = config.getMaxSamples();
        if (maxSamples < size) { cstats << "Number of samples: " << maxSamples << '\n'; }
    }
    cstats << "Number of variables: " << framework.varSize() << '\n';
    cstats << std::string(60, '-') << '\n';
}

void Framework::Expand::printStats(Explanation const & explanation, Network::Dataset const & data,
                                   ExplanationIdx idx) const {
    auto & print = framework.getPrint();
    assert(not print.ignoringStats());
    auto & cstats = print.stats();
    auto const defaultPrecision = cstats.precision();

    std::size_t const varSize = framework.varSize();
    std::size_t const expVarSize = explanation.varSize();
    assert(expVarSize <= varSize);

    std::size_t const dataSize = data.size();
    auto const & sample = data.getSample(idx);
    auto const & expClass = data.getExpectedClassification(idx).label;
    auto const & compClass = data.getComputedOutput(idx).classification.label;

    std::size_t const fixedCount = explanation.getFixedCount();
    assert(fixedCount <= expVarSize);

    std::size_t const termSize = explanation.termSize();
    assert(termSize > 0);

    cstats << '\n';
    cstats << "sample [" << idx + 1 << '/' << dataSize << "]: " << sample << '\n';
    cstats << "expected output: " << expClass << '\n';
    cstats << "computed output: " << compClass << '\n';
    cstats << "#checks: " << verifierPtr->getChecksCount() << '\n';
    cstats << "#features: " << expVarSize << '/' << varSize << std::endl;

    assert(not explanation.supportsVolume() or explanation.getRelativeVolumeSkipFixed() > 0);
    assert(explanation.getRelativeVolumeSkipFixed() <= 1);
    assert((explanation.getRelativeVolumeSkipFixed() < 1) == (fixedCount < expVarSize));

    bool const isAbductive = (fixedCount == expVarSize);
    if (isAbductive) {
        assert(termSize == expVarSize);
        assert(explanation.supportsVolume());
        assert(explanation.getRelativeVolumeSkipFixed() == 1);
        return;
    }

    cstats << "#fixed features: " << fixedCount << '/' << varSize << '\n';
    cstats << "#terms: " << termSize << std::endl;

    if (not explanation.supportsVolume()) { return; }

    Float const relVolume = explanation.getRelativeVolumeSkipFixed();

    cstats << "relVolume*: " << std::setprecision(1) << (relVolume * 100) << "%" << std::setprecision(defaultPrecision)
           << std::endl;
}
} // namespace xspace
