#ifndef SPEXPLAIN_FRAMEWORK_H
#define SPEXPLAIN_FRAMEWORK_H

#include <spexplain/common/Interval.h>
#include <spexplain/common/Var.h>
#include <spexplain/network/Network.h>

#include <cassert>
#include <iosfwd>
#include <memory>
#include <string_view>
#include <vector>

namespace spexplain {
class Explanation;

using Sample = Network::Sample;

using Explanations = std::vector<std::unique_ptr<Explanation>>;

static_assert(std::is_same_v<Sample::Idx, Explanations::size_type>);

inline std::unique_ptr<Explanation> & getExplanationPtr(Explanations & explanations, Sample::Idx idx) {
    return explanations[idx];
}
inline Explanation & getExplanation(Explanations & explanations, Sample::Idx idx) {
    return *getExplanationPtr(explanations, idx);
}

// Class that represents the Space Explanation Framework
class Framework {
public:
    class Config;

    class Expand;

    // Not inline because of fwd-decl. types
    Framework();
    Framework(Config const &);
    Framework(Config const &, std::unique_ptr<Network>);
    Framework(Config const &, std::unique_ptr<Network>, std::istream & expandStrategiesSpec);
    ~Framework();

    void setConfig(Config const &);

    Config const & getConfig() const {
        assert(configPtr);
        return *configPtr;
    }

    void setNetwork(std::unique_ptr<Network>);

    Network const & getNetwork() const {
        assert(networkPtr);
        return *networkPtr;
    }

    void setExpand(std::istream & strategiesSpec);

    // Sufficient to set in Config
    void setVerifierName(std::string_view name);

    // Sufficient to set in Config
    void setExplanationsFileName(std::string_view fileName);
    void setStatsFileName(std::string_view fileName);
    void setTimesFileName(std::string_view fileName);

    std::size_t varSize() const { return varNames.size(); }
    VarName const & getVarName(VarIdx idx) const { return varNames[idx]; }

    Interval const & getDomainInterval(VarIdx idx) const { return domainIntervals[idx]; }

    void dumpDomainsAsSmtLib2Query();
    void dumpClassificationsAsSmtLib2Queries();

    Explanations explain(Network::Dataset &);

    // Allows further expansion of explanations in a file
    Explanations expand(std::string_view fileName, Network::Dataset &);

    // Allows further expansion of already existing explanations
    void expand(Explanations &, Network::Dataset const &);

protected:
    friend class PartialExplanation;

    class Preprocess;
    class Parse;

    class Print;

    using VarNames = std::vector<VarName>;

    static_assert(std::is_same_v<VarNames::size_type, VarIdx>);

    Expand const & getExpand() const {
        assert(expandPtr);
        return *expandPtr;
    }

    Preprocess const & getPreprocess() const {
        assert(preprocessPtr);
        return *preprocessPtr;
    }

    Print const & getPrint() const {
        assert(printPtr);
        return *printPtr;
    }

    std::unique_ptr<Config const> configPtr;

    std::unique_ptr<Network> networkPtr{};
    VarNames varNames{};
    std::vector<Interval> domainIntervals{};

    std::unique_ptr<Expand> expandPtr{};

    std::unique_ptr<Preprocess> preprocessPtr{};

    std::unique_ptr<Print> printPtr{};
};
} // namespace spexplain

#endif // SPEXPLAIN_FRAMEWORK_H
