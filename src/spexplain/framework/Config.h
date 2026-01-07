#ifndef SPEXPLAIN_CONFIG_H
#define SPEXPLAIN_CONFIG_H

#include "Framework.h"
#include "explanation/IntervalExplanation.h"

#include <spexplain/network/Containers.h>
#include <spexplain/network/Dataset.h>

#include <cassert>
#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace spexplain {
//+ move parsing cmdline options here
class Framework::Config {
public:
    using Verbosity = short;

    enum class DefaultSampleNeuronActivations { none, all, active, inactive };

    struct HiddenNeuronPosition {
        std::size_t layer;
        std::size_t node;
        bool negated{false};
    };

    // Not always, it may use Marabou if suitable:
    // static inline std::string const defaultVerifierName = "opensmt";
    static inline std::string const defaultExplanationsFileName = "phi.txt";

    void init(Network const &) noexcept;

    Network const & getNetwork() const {
        assert(networkPtr);
        return *networkPtr;
    }

    void setVerifierName(std::string_view name) { verifierName = name; }

    void setExplanationsFileName(std::string_view fileName) { explanationsFileName = fileName; }
    void setStatsFileName(std::string_view fileName) { statsFileName = fileName; }
    void setTimesFileName(std::string_view fileName) { timesFileName = fileName; }

    void setVerbosity(Verbosity verb) { verbosity = verb; }
    void beVerbose() { setVerbosity(1); }
    void beQuiet() { setVerbosity(-1); }

    void reverseVarOrdering() { reverseVarOrder = true; }

    void fixDefaultSampleNeuronActivations(
        DefaultSampleNeuronActivations sampleNeuronActivations = DefaultSampleNeuronActivations::all) {
        _fixDefaultSampleNeuronActivations = sampleNeuronActivations;
    }
    void preferDefaultSampleNeuronActivations(
        DefaultSampleNeuronActivations sampleNeuronActivations = DefaultSampleNeuronActivations::all) {
        _preferDefaultSampleNeuronActivations = sampleNeuronActivations;
    }
    void fixAllSampleNeuronActivationsAt(HiddenNeuronPosition const & pos) {
        fixAllSampleNeuronActivationsMap.insertOrAssign(pos.layer, pos.node, not pos.negated);
    }
    void preferAllSampleNeuronActivationsAt(HiddenNeuronPosition const & pos) {
        preferAllSampleNeuronActivationsMap.insertOrAssign(pos.layer, pos.node, not pos.negated);
    }
    void fixSampleNeuronActivationAt(Sample::Idx, HiddenNeuronPosition const &);
    void preferSampleNeuronActivationAt(Sample::Idx, HiddenNeuronPosition const &);

    void setPrintIntervalExplanationsFormat(IntervalExplanation::PrintFormat tp) {
        intervalExplanationPrintFormat = tp;
    }
    void printIntervalExplanationsInSmtLib2Format() {
        setPrintIntervalExplanationsFormat(IntervalExplanation::PrintFormat::smtlib2);
    }
    void printIntervalExplanationsInBoundFormat() {
        setPrintIntervalExplanationsFormat(IntervalExplanation::PrintFormat::bounds);
    }
    void printIntervalExplanationsInIntervalFormat() {
        setPrintIntervalExplanationsFormat(IntervalExplanation::PrintFormat::intervals);
    }

    void shuffleSamples() { _shuffleSamples = true; }

    void setMaxSamples(std::size_t n) { maxSamples = n; }
    void setFirstSampleIdx(std::size_t idx) { firstSample = idx; }
    void setLastSampleIdx(std::size_t idx) { lastSample = idx; }

    void filterCorrectSamples() { optFilterCorrectSamples = true; }
    void filterIncorrectSamples() { optFilterCorrectSamples = false; }
    void filterSamplesOfExpectedClass(Network::Classification::Label l) { optFilterSamplesOfExpectedClass = l; }

    void setTimeLimitPerExplanation(std::size_t limit_ms) {
        timeLimitPerExplanation = std::chrono::milliseconds{limit_ms};
    }

    [[nodiscard]]
    std::string_view getVerifierName() const {
        return verifierName;
    }
    [[nodiscard]]
    bool verifierNameIsSet() const {
        return not getVerifierName().empty();
    }

    [[nodiscard]]
    std::string_view getExplanationsFileName() const {
        if (not explanationsFileNameIsSet()) { return defaultExplanationsFileName; }
        return explanationsFileName;
    }
    [[nodiscard]]
    bool explanationsFileNameIsSet() const {
        return not explanationsFileName.empty();
    }
    [[nodiscard]]
    std::string_view getStatsFileName() const {
        return statsFileName;
    }
    [[nodiscard]]
    bool statsFileNameIsSet() const {
        return not getStatsFileName().empty();
    }
    [[nodiscard]]
    std::string_view getTimesFileName() const {
        return timesFileName;
    }
    [[nodiscard]]
    bool timesFileNameIsSet() const {
        return not getTimesFileName().empty();
    }

    [[nodiscard]]
    Verbosity getVerbosity() const {
        return verbosity;
    }
    [[nodiscard]]
    bool isVerbose() const {
        return getVerbosity() > 0;
    }
    [[nodiscard]]
    bool isQuiet() const {
        return getVerbosity() < 0;
    }

    [[nodiscard]]
    bool isReverseVarOrdering() const {
        return reverseVarOrder;
    }

    [[nodiscard]]
    DefaultSampleNeuronActivations getDefaultFixingOfSampleNeuronActivations() const {
        return _fixDefaultSampleNeuronActivations;
    }
    [[nodiscard]]
    DefaultSampleNeuronActivations getDefaultPreferenceOfSampleNeuronActivations() const {
        return _preferDefaultSampleNeuronActivations;
    }
    [[nodiscard]]
    std::optional<bool> tryGetFixingOfAllSampleNeuronActivationsAt(std::size_t layer, std::size_t node) const {
        return fixAllSampleNeuronActivationsMap.tryGetAt(layer, node);
    }
    [[nodiscard]]
    std::optional<bool> tryGetPreferenceOfAllSampleNeuronActivationsAt(std::size_t layer, std::size_t node) const {
        return preferAllSampleNeuronActivationsMap.tryGetAt(layer, node);
    }
    [[nodiscard]]
    std::optional<bool> tryGetFixingOfSampleNeuronActivationAt(Sample::Idx, std::size_t layer, std::size_t node) const;
    [[nodiscard]]
    std::optional<bool> tryGetPreferenceOfSampleNeuronActivationAt(Sample::Idx, std::size_t layer,
                                                                   std::size_t node) const;

    [[nodiscard]]
    IntervalExplanation::PrintFormat const & getPrintingIntervalExplanationsFormat() const {
        return intervalExplanationPrintFormat;
    }
    [[nodiscard]]
    bool printingIntervalExplanationsInSmtLib2Format() const {
        return intervalExplanationPrintFormat == IntervalExplanation::PrintFormat::smtlib2;
    }
    [[nodiscard]]
    bool printingIntervalExplanationsInBoundFormat() const {
        return intervalExplanationPrintFormat == IntervalExplanation::PrintFormat::bounds;
    }
    [[nodiscard]]
    bool printingIntervalExplanationsInIntervalFormat() const {
        return intervalExplanationPrintFormat == IntervalExplanation::PrintFormat::intervals;
    }
    [[nodiscard]]
    char getPrintingIntervalExplanationsDelim() const {
        switch (getPrintingIntervalExplanationsFormat()) {
            default:
                return IntervalExplanation::PrintConfig{}.delim;
            case IntervalExplanation::PrintFormat::bounds:
                return IntervalExplanation::defaultBoundsPrintConfig.delim;
            case IntervalExplanation::PrintFormat::intervals:
                return IntervalExplanation::defaultIntervalsPrintConfig.delim;
        }
    }

    [[nodiscard]]
    bool shufflingSamples() const {
        return _shuffleSamples;
    }

    [[nodiscard]]
    std::size_t getMaxSamples() const {
        return maxSamples;
    }
    [[nodiscard]]
    bool limitingMaxSamples() const {
        return getMaxSamples() > 0;
    }
    [[nodiscard]]
    std::size_t getFirstSampleIdx() const {
        return firstSample;
    }
    [[nodiscard]]
    bool limitingFirstSample() const {
        return getFirstSampleIdx() > 0;
    }
    [[nodiscard]]
    std::size_t getLastSampleIdx() const {
        return lastSample;
    }
    [[nodiscard]]
    bool limitingLastSample() const {
        return getLastSampleIdx() > 0;
    }

    [[nodiscard]]
    bool filteringCorrectSamples() const {
        return optFilterCorrectSamples.has_value() and *optFilterCorrectSamples;
    }
    [[nodiscard]]
    bool filteringIncorrectSamples() const {
        return optFilterCorrectSamples.has_value() and not *optFilterCorrectSamples;
    }
    [[nodiscard]]
    bool filteringSamplesOfExpectedClass() const {
        return optFilterSamplesOfExpectedClass.has_value();
    }
    [[nodiscard]]
    Network::Classification::Label const & getSamplesExpectedClassFilter() const {
        assert(filteringSamplesOfExpectedClass());
        return *optFilterSamplesOfExpectedClass;
    }

    [[nodiscard]]
    std::chrono::milliseconds getTimeLimitPerExplanation() const {
        return timeLimitPerExplanation;
    }
    [[nodiscard]]
    bool timeLimitPerExplanationIsSet() const {
        return getTimeLimitPerExplanation().count() > 0;
    }

protected:
    Network const * networkPtr{};

    std::string_view verifierName{};

    std::string_view explanationsFileName{};
    std::string_view statsFileName{};
    std::string_view timesFileName{};

    Verbosity verbosity{};

    bool reverseVarOrder{};

    DefaultSampleNeuronActivations _fixDefaultSampleNeuronActivations{DefaultSampleNeuronActivations::none};
    DefaultSampleNeuronActivations _preferDefaultSampleNeuronActivations{DefaultSampleNeuronActivations::all};
    NetworkMap<bool> fixAllSampleNeuronActivationsMap{};
    NetworkMap<bool> preferAllSampleNeuronActivationsMap{};
    std::unordered_map<Sample::Idx, NetworkMap<bool>> fixSampleNeuronActivationMaps{};
    std::unordered_map<Sample::Idx, NetworkMap<bool>> preferSampleNeuronActivationMaps{};

    IntervalExplanation::PrintFormat intervalExplanationPrintFormat{IntervalExplanation::PrintFormat::bounds};

    bool _shuffleSamples{};

    std::size_t maxSamples{};
    std::size_t firstSample{};
    std::size_t lastSample{};

    std::optional<bool> optFilterCorrectSamples{};
    std::optional<Network::Classification::Label> optFilterSamplesOfExpectedClass{};

    std::chrono::milliseconds timeLimitPerExplanation{};
};

Framework::Config::DefaultSampleNeuronActivations makeDefaultSampleNeuronActivations(std::string_view);

Framework::Config::HiddenNeuronPosition makeHiddenNeuronPosition(std::string_view);
std::pair<Sample::Idx, Framework::Config::HiddenNeuronPosition> makeHiddenNeuronPositionOfSample(std::string_view);

bool usingSampleNeuronActivations(Framework::Config::DefaultSampleNeuronActivations, bool sampleActivated);

void parseFixSampleNeuronActivations(Framework::Config &, std::string_view fileName);
void parsePreferSampleNeuronActivations(Framework::Config &, std::string_view fileName);
} // namespace spexplain

#endif // SPEXPLAIN_CONFIG_H
