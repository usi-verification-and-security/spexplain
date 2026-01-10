#include "Expand.h"

#include "../Config.h"
#include "../Preprocess.h"
#include "../Print.h"
#include "../explanation/Explanation.h"
#include "strategy/Factory.h"
#include "strategy/Strategies.h"

#include <spexplain/common/Core.h>
#include <spexplain/common/String.h>

#include <verifiers/Verifier.h>
#include <verifiers/opensmt/OpenSMTVerifier.h>
#ifdef MARABOU
#include <verifiers/marabou/MarabouVerifier.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

namespace spexplain {
Framework::Expand::Expand(Framework & fw) : framework{fw} {}

void Framework::Expand::setStrategies() {
    auto strategyPtr = std::make_unique<expand::opensmt::InterpolationStrategy>(*this);
    addStrategy(std::move(strategyPtr));
}

void Framework::Expand::setStrategies(std::istream & is) {
    // pipe character '|' reserved for disjunctions
    static constexpr char strategyDelim = ';';

    is >> std::ws;
    if (is.eof()) { return setStrategies(); }
    assert(is.good());

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
    auto const & config = framework.getConfig();
    auto verifierName = config.getVerifierName();

    setVerifier(verifierName);
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

    assert(not config.limitingLastSample() || config.limitingFirstSample());
    if (config.limitingFirstSample()) {
        auto const firstIdx = config.getFirstSampleIdx() - 1;
        assert(firstIdx < indices.size());
        auto const firstIt = indices.begin() + firstIdx;
        auto const lastIdx_ = config.limitingLastSample() ? config.getLastSampleIdx() : indices.size();
        assert(lastIdx_ <= indices.size());
        auto const lastIt = indices.begin() + lastIdx_;
        indices = decltype(indices)(firstIt, lastIt);
    }

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

void Framework::Expand::dumpDomainsAsSmtLib2Query() {
    auto & print = framework.getPrint();
    auto & cinfo = print.info();

    std::string const fname = "psi_d.smt2";
    std::ofstream ofs{fname};

    printDomainsAsSmtLib2Query(ofs);

    cinfo << "Dumped domains SMT query to: " << fname << '\n';
}

void Framework::Expand::dumpClassificationsAsSmtLib2Queries() {
    auto & print = framework.getPrint();
    auto & cinfo = print.info();

    auto & network = framework.getNetwork();
    std::size_t const nClasses = network.nClasses();

    for (Network::Classification::Label l = 0; l < nClasses; ++l) {
        Network::Classification cls{.label = l};

        std::string const fname = "psi_c" + std::to_string(l) + ".smt2";
        std::ofstream ofs{fname};

        printClassificationAsSmtLib2Query(ofs, cls);

        cinfo << "Dumped classification " << l << " SMT query to: " << fname << '\n';
    }
}

void Framework::Expand::printDomainsAsSmtLib2Query(std::ostream & os) {
    auto & verifier = getVerifier();

    initVerifier();

    assertGroundModel();
    assertSampleModel();

    verifier.printSmtLib2Query(os);
}

void Framework::Expand::printClassificationAsSmtLib2Query(std::ostream & os, Network::Classification const & cls) {
    auto & verifier = getVerifier();

    auto & label = cls.label;

    initVerifier();

    os << "(set-info :source \"class " << label << "\")\n";
    assertGroundModel();
    assertSampleModel();
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

    auto const & config = framework.getConfig();
    bool const timeoutPerIsSet = config.timeLimitPerExplanationIsSet();
    [[maybe_unused]]
    auto const timeoutPer = config.getTimeLimitPerExplanation();

    auto & print = framework.getPrint();
    bool const printingInfo = not print.ignoringInfo();
    bool const printingExplanations = not print.ignoringExplanations();
    bool const printingStats = not print.ignoringStats();
    bool const printingTimes = not print.ignoringTimes();
    auto & cinfo = print.info();
    auto & cexp = print.explanations();
    auto & cstats = print.stats();
    auto & ctimes = print.times();
    assert(printingExplanations);

    if (printingInfo) {
        cinfo << "Writing explanations to: " << config.getExplanationsFileName() << '\n';
        if (printingStats) { cinfo << "Writing statistics to: " << config.getStatsFileName() << '\n'; }
        if (printingTimes) { cinfo << "Writing runtimes per explanation to: " << config.getTimesFileName() << '\n'; }
        cinfo << '\n';
        printHead(cinfo, data);
    }

    if (printingStats) { printHead(cstats, data); }

    auto const startTimeF = [printingTimes]() -> std::chrono::time_point<std::chrono::steady_clock> {
        if (not printingTimes) { return {}; }
        return std::chrono::steady_clock::now();
    };

    initVerifier();
    preprocessGroundModel(data);
    assertGroundModel();

    Network::Dataset::SampleIndices const indices = makeSampleIndices(data);
    for (auto idx : indices) {
        [[maybe_unused]]
        auto const start = startTimeF();

        if (printingInfo) {
            printProgress(cinfo, data, idx);
            cinfo << " ... ";
            cinfo.flush();
        }

        bool timeout = false;
        if (timeoutPerIsSet) { verifierPtr->setTimeLimit(timeoutPer); }

        auto const & output = data.getComputedOutput(idx);

        preprocessSampleModel(idx, output);
        assertSampleModel();

        auto const & cls = output.classification;
        assertClassification(cls);

        try {
            for (auto & strategy : strategies) {
                strategy->execute(explanations, data, idx);
            }
        } catch (UnknownResultInternalException) { timeout = true; }

        assert(timeoutPerIsSet or not timeout);

        if (not timeout) {
            postprocessExplanation(explanations, idx);

            auto & explanation = getExplanation(explanations, idx);
            cinfo << "done";
            //+ get rid of the conditionals
            if (printingStats) { printStatsOf(explanation, data, idx); }
            if (printingExplanations) {
                explanation.print(cexp);
                cexp << std::endl;
            }
        } else {
            cinfo << "timeout";
            if (printingStats) {
                printStatsHeadOf(data, idx);
                cstats << "<timeout>\n";
            }
            if (printingExplanations) {
                //! the default format does not work if not yielding interval explanations
                char const delim = config.getPrintingIntervalExplanationsDelim();
                cexp << invalidExplanationString << delim << std::endl;
            }
        }
        cinfo << std::endl;

        resetClassification();

        resetSampleModel();

        if (not printingTimes) { continue; }

        if (not timeout) {
            auto const finish = std::chrono::steady_clock::now();
            std::chrono::duration<double> const duration = finish - start;
            ctimes << std::setprecision(3) << duration.count();
        } else {
            ctimes << invalidExplanationString;
        }
        ctimes << std::endl;
    }

    resetGroundModel();

    // resetVerifier() not needed here

    cinfo << "\nDone." << std::endl;
}

void Framework::Expand::initVerifier() {
    assert(verifierPtr);
    auto & network = framework.getNetwork();
    verifierPtr->init(network);
}

void Framework::Expand::preprocessGroundModel(Network::Dataset const &) {
    assert(verifierPtr);
}

void Framework::Expand::preprocessSampleModel(Sample::Idx idx, Network::Output const & output) {
    assert(verifierPtr);
    auto & verifier = *verifierPtr;
    auto & network = framework.getNetwork();

    using DefaultSampleNeuronActivations = Config::DefaultSampleNeuronActivations;
    auto const & config = framework.getConfig();
    DefaultSampleNeuronActivations const defaultFixingOfSampleNeuronActivations =
        config.getDefaultFixingOfSampleNeuronActivations();
    DefaultSampleNeuronActivations const defaultPreferenceOfSampleNeuronActivations =
        config.getDefaultPreferenceOfSampleNeuronActivations();

    xai::verifiers::LayerIndex const nHiddenLayers = network.nHiddenLayers();
    assert(nHiddenLayers == network.nLayers() - 2);
    for (xai::verifiers::LayerIndex layer = 1; layer < nHiddenLayers + 1; ++layer) {
        xai::verifiers::NodeIndex const nNodes = network.getLayerSize(layer);
        for (xai::verifiers::NodeIndex node = 0; node < nNodes; ++node) {
            bool const activated = activatedHiddenNeuron(output, layer, node);

            if (auto optFixOne = config.tryGetFixingOfSampleNeuronActivationAt(idx, layer, node)) {
                if (*optFixOne) { verifier.fixNeuronActivation(layer, node, activated); }
            } else if (auto optFixAll = config.tryGetFixingOfAllSampleNeuronActivationsAt(layer, node)) {
                if (*optFixAll) { verifier.fixNeuronActivation(layer, node, activated); }
            } else if (usingSampleNeuronActivations(defaultFixingOfSampleNeuronActivations, activated)) {
                verifier.tryFixNeuronActivation(layer, node, activated);
            }

            if (auto optPreferOne = config.tryGetPreferenceOfSampleNeuronActivationAt(idx, layer, node)) {
                if (*optPreferOne) { verifier.preferNeuronActivation(layer, node, activated); }
            } else if (auto optPreferAll = config.tryGetPreferenceOfAllSampleNeuronActivationsAt(layer, node)) {
                if (*optPreferAll) { verifier.preferNeuronActivation(layer, node, activated); }
            } else if (usingSampleNeuronActivations(defaultPreferenceOfSampleNeuronActivations, activated)) {
                verifier.tryPreferNeuronActivation(layer, node, activated);
            }
        }
    }
}

void Framework::Expand::assertGroundModel() {
    assert(verifierPtr);
    verifierPtr->assertGroundModel();
}

void Framework::Expand::assertSampleModel() {
    assert(verifierPtr);
    verifierPtr->assertSampleModel();
}

void Framework::Expand::resetSampleModel() {
    assert(verifierPtr);
    verifierPtr->resetSampleModel();
}

void Framework::Expand::resetGroundModel() {
    assert(verifierPtr);
    verifierPtr->resetGroundModel();
}

void Framework::Expand::resetVerifier() {
    assert(verifierPtr);
    verifierPtr->reset();
}

void Framework::Expand::assertClassification(Network::Classification const & cls) {
    verifierPtr->push();

    auto & network = framework.getNetwork();
    auto const outputLayerIndex = network.nLayers() - 1;

    auto const & label = cls.label;

    assert(network.nClasses() >= 2);
    if (network.nClasses() > 2) {
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

void Framework::Expand::postprocessExplanation(Explanations & explanations, Sample::Idx idx) {
    auto & lastStrategy = getLastStrategy();
    auto & explanationPtr = getExplanationPtr(explanations, idx);

    if (auto cexplanationPtr = verifierPtr->getSampleModelRestrictions(framework)) {
        lastStrategy.intersectExplanation(explanationPtr, std::move(cexplanationPtr));
    }
}

void Framework::Expand::printHead(std::ostream & os, Network::Dataset const & data) const {
    auto const & config = framework.getConfig();

    std::size_t const size = data.size();
    os << "Dataset size: " << size << '\n';
    os << "Number of variables: " << framework.varSize() << '\n';

    if (config.shufflingSamples()) { os << "Shuffled samples\n"; }
    if (config.limitingFirstSample()) {
        auto const firstIdx = config.getFirstSampleIdx();
        if (firstIdx > 1) { os << "Starting from sample " << firstIdx << '\n'; }
    }
    if (config.limitingLastSample()) {
        auto const lastIdx = config.getLastSampleIdx();
        if (lastIdx < size) { os << "Ending at sample " << lastIdx << '\n'; }
    }
    if (config.limitingMaxSamples()) {
        auto const maxSamples = config.getMaxSamples();
        if (maxSamples < size) { os << "Limited number of samples: " << maxSamples << '\n'; }
    }

    if (config.timeLimitPerExplanationIsSet()) {
        auto const timeLimitPer_ms = config.getTimeLimitPerExplanation().count();
        double const timeLimitPer_s = static_cast<double>(timeLimitPer_ms) / 1000;
        os << "Timeout per sample [s]: " << timeLimitPer_s << '\n';
    }

    os << std::string(60, '-') << std::endl;
}

void Framework::Expand::printProgress(std::ostream & os, Network::Dataset const & data, Sample::Idx idx,
                                      std::string_view caption) const {
    std::size_t const dataSize = data.size();
    os << caption << " [" << idx + 1 << '/' << dataSize << ']';
}

void Framework::Expand::printStatsOf(Explanation const & explanation, Network::Dataset const & data,
                                     Sample::Idx idx) const {

    printStatsHeadOf(data, idx);
    printStatsBodyOf(explanation);
}

void Framework::Expand::printStatsHeadOf(Network::Dataset const & data, Sample::Idx idx) const {
    auto & print = framework.getPrint();
    assert(not print.ignoringStats());
    auto & cstats = print.stats();

    auto const & sample = data.getSample(idx);
    auto const & expClass = data.getExpectedClassification(idx).label;
    auto const & compClass = data.getComputedOutput(idx).classification.label;

    cstats << '\n';
    printProgress(cstats, data, idx);
    cstats << ": " << sample << '\n';
    cstats << "expected output: " << expClass << '\n';
    cstats << "computed output: " << compClass << '\n';
    cstats << "#checks: " << verifierPtr->getChecksCount() << '\n';
}

void Framework::Expand::printStatsBodyOf(Explanation const & explanation) const {
    auto & print = framework.getPrint();
    assert(not print.ignoringStats());
    auto & cstats = print.stats();
    auto const defaultPrecision = cstats.precision();

    std::size_t const varSize = framework.varSize();
    std::size_t const expVarSize = explanation.varSize();
    assert(expVarSize <= varSize);

    std::size_t const fixedCount = explanation.getFixedCount();
    assert(fixedCount <= expVarSize);

    std::size_t const termSize = explanation.termSize();
    assert(termSize > 0);

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
} // namespace spexplain
