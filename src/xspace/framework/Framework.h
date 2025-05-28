#ifndef XSPACE_FRAMEWORK_H
#define XSPACE_FRAMEWORK_H

#include <xspace/common/Interval.h>
#include <xspace/common/Var.h>
#include <xspace/network/Network.h>

#include <cassert>
#include <iosfwd>
#include <memory>
#include <string_view>
#include <vector>

namespace xspace {
class Explanation;

using Explanations = std::vector<std::unique_ptr<Explanation>>;
using ExplanationIdx = Explanations::size_type;

inline std::unique_ptr<Explanation> & getExplanationPtr(Explanations & explanations, ExplanationIdx idx) {
    return explanations[idx];
}
inline Explanation & getExplanation(Explanations & explanations, ExplanationIdx idx) {
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
    Framework(Config const &, std::unique_ptr<Network>, std::string_view verifierName,
              std::istream & expandStrategiesSpec);
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
    void setExpand(std::string_view verifierName, std::istream & strategiesSpec);

    std::size_t varSize() const { return varNames.size(); }
    VarName const & getVarName(VarIdx idx) const { return varNames[idx]; }

    Interval const & getDomainInterval(VarIdx idx) const { return domainIntervals[idx]; }

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

    std::unique_ptr<Config> configPtr;

    std::unique_ptr<Network> networkPtr{};
    VarNames varNames{};
    std::vector<Interval> domainIntervals{};

    std::unique_ptr<Expand> expandPtr{};

    std::unique_ptr<Preprocess> preprocessPtr{};

    std::unique_ptr<Print> printPtr{};
};
} // namespace xspace

#endif // XSPACE_FRAMEWORK_H
