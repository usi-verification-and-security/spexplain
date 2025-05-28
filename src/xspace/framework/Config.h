#ifndef XSPACE_CONFIG_H
#define XSPACE_CONFIG_H

#include "Framework.h"
#include "explanation/IntervalExplanation.h"

#include <xspace/network/Dataset.h>

#include <optional>

namespace xspace {
//+ move parsing cmdline options here
//+ after construction, make the conf const
class Framework::Config {
public:
    using Verbosity = short;

    void setVerbosity(Verbosity verb) { verbosity = verb; }
    void beVerbose() { setVerbosity(1); }

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

    Verbosity getVerbosity() const { return verbosity; }
    bool isVerbose() const { return getVerbosity() > 0; }

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
    Verbosity verbosity{};

    bool reverseVarOrder{};

    IntervalExplanation::PrintFormat intervalExplanationPrintFormat{IntervalExplanation::PrintFormat::bounds};

    bool _shuffleSamples{};

    std::size_t maxSamples{};

    std::optional<bool> optFilterCorrectSamples{};
    std::optional<Network::Classification::Label> optFilterSamplesOfExpectedClass{};
};
} // namespace xspace

#endif // XSPACE_CONFIG_H
