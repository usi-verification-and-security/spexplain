#ifndef SPEXPLAIN_CONFIG_H
#define SPEXPLAIN_CONFIG_H

#include "Framework.h"
#include "explanation/IntervalExplanation.h"

#include <spexplain/network/Dataset.h>

#include <optional>
#include <string>
#include <string_view>

namespace spexplain {
//+ move parsing cmdline options here
class Framework::Config {
public:
    using Verbosity = short;

    // Not always, it may use Marabou if suitable:
    // static inline std::string const defaultVerifierName = "opensmt";
    static inline std::string const defaultExplanationsFileName = "phi.txt";

    void setVerifierName(std::string_view name) { verifierName = name; }

    void setExplanationsFileName(std::string_view fileName) { explanationsFileName = fileName; }
    void setStatsFileName(std::string_view fileName) { statsFileName = fileName; }
    void setTimesFileName(std::string_view fileName) { timesFileName = fileName; }

    void setVerbosity(Verbosity verb) { verbosity = verb; }
    void beVerbose() { setVerbosity(1); }
    void beQuiet() { setVerbosity(-1); }

    void reverseVarOrdering() { reverseVarOrder = true; }

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

    void filterCorrectSamples() { optFilterCorrectSamples = true; }
    void filterIncorrectSamples() { optFilterCorrectSamples = false; }
    void filterSamplesOfExpectedClass(Network::Classification::Label l) { optFilterSamplesOfExpectedClass = l; }

    std::string_view getVerifierName() const { return verifierName; }
    bool verifierNameIsSet() const { return not getVerifierName().empty(); }

    std::string_view getExplanationsFileName() const {
        if (not explanationsFileNameIsSet()) { return defaultExplanationsFileName; }
        return explanationsFileName;
    }
    bool explanationsFileNameIsSet() const { return not explanationsFileName.empty(); }
    std::string_view getStatsFileName() const { return statsFileName; }
    bool statsFileNameIsSet() const { return not getStatsFileName().empty(); }
    std::string_view getTimesFileName() const { return timesFileName; }
    bool timesFileNameIsSet() const { return not getTimesFileName().empty(); }

    Verbosity getVerbosity() const { return verbosity; }
    bool isVerbose() const { return getVerbosity() > 0; }
    bool isQuiet() const { return getVerbosity() < 0; }

    bool isReverseVarOrdering() const { return reverseVarOrder; }

    IntervalExplanation::PrintFormat const & getPrintingIntervalExplanationsFormat() const {
        return intervalExplanationPrintFormat;
    }
    bool printingIntervalExplanationsInSmtLib2Format() const {
        return intervalExplanationPrintFormat == IntervalExplanation::PrintFormat::smtlib2;
    }
    bool printingIntervalExplanationsInBoundFormat() const {
        return intervalExplanationPrintFormat == IntervalExplanation::PrintFormat::bounds;
    }
    bool printingIntervalExplanationsInIntervalFormat() const {
        return intervalExplanationPrintFormat == IntervalExplanation::PrintFormat::intervals;
    }

    bool shufflingSamples() const { return _shuffleSamples; }

    std::size_t getMaxSamples() const { return maxSamples; }
    bool limitingMaxSamples() const { return getMaxSamples() > 0; }

    bool filteringCorrectSamples() const { return optFilterCorrectSamples.has_value() and *optFilterCorrectSamples; }
    bool filteringIncorrectSamples() const {
        return optFilterCorrectSamples.has_value() and not *optFilterCorrectSamples;
    }
    bool filteringSamplesOfExpectedClass() const { return optFilterSamplesOfExpectedClass.has_value(); }
    Network::Classification::Label const & getSamplesExpectedClassFilter() const {
        assert(filteringSamplesOfExpectedClass());
        return *optFilterSamplesOfExpectedClass;
    }

protected:
    std::string_view verifierName{};

    std::string_view explanationsFileName{};
    std::string_view statsFileName{};
    std::string_view timesFileName{};

    Verbosity verbosity{};

    bool reverseVarOrder{};

    IntervalExplanation::PrintFormat intervalExplanationPrintFormat{IntervalExplanation::PrintFormat::bounds};

    bool _shuffleSamples{};

    std::size_t maxSamples{};

    std::optional<bool> optFilterCorrectSamples{};
    std::optional<Network::Classification::Label> optFilterSamplesOfExpectedClass{};
};
} // namespace spexplain

#endif // SPEXPLAIN_CONFIG_H
