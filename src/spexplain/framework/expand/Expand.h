#ifndef SPEXPLAIN_EXPAND_H
#define SPEXPLAIN_EXPAND_H

#include "../Framework.h"

#include <spexplain/common/Var.h>
#include <spexplain/network/Dataset.h>

#include <memory>
#include <vector>

namespace xai::verifiers {
class Verifier;
}

namespace spexplain {
class Explanation;

static_assert(std::is_same_v<ExplanationIdx, Network::Sample::Idx>);

//! rename to `Explain`?
class Framework::Expand {
public:
    struct VarOrdering {
        enum class Type { regular, reverse, manual };

        Type type{Type::regular};
        std::vector<VarIdx> order{};
    };

    class Strategy;
    class NopStrategy;
    class AbductiveStrategy;
    class TrialAndErrorStrategy;
    class UnsatCoreStrategy;
    //! does not expand, it shrinks
    class SliceStrategy;

    using Strategies = std::vector<std::unique_ptr<Strategy>>;

    static constexpr char const * invalidExplanationString = "<null>";

    Expand(Framework &);

    Framework const & getFramework() const { return framework; }

    // Using default strategies
    void setStrategies();
    void setStrategies(std::istream &);

    // Based on Config
    void setVerifier();
    void setVerifier(std::string_view name);

    xai::verifiers::Verifier const & getVerifier() const {
        assert(verifierPtr);
        return *verifierPtr;
    }

    void dumpDomainsAsSmtLib2Query();
    void dumpClassificationsAsSmtLib2Queries();
    void printDomainsAsSmtLib2Query(std::ostream &);
    void printClassificationAsSmtLib2Query(std::ostream &, Network::Classification const &);

    void operator()(Explanations &, Network::Dataset const &);

protected:
    struct UnknownResultInternalException {};

    void addStrategy(std::unique_ptr<Strategy>);

    std::unique_ptr<xai::verifiers::Verifier> makeVerifier(std::string_view name) const;
    void setVerifier(std::unique_ptr<xai::verifiers::Verifier>);

    Network::Dataset::SampleIndices makeSampleIndices(Network::Dataset const &) const;

    void initVerifier();

    void assertModel();
    void resetModel();

    void assertClassification(Network::Classification const &);
    void resetClassification();

    void printHead(std::ostream &, Network::Dataset const &) const;
    void printProgress(std::ostream &, Network::Dataset const &, ExplanationIdx,
                       std::string_view caption = "sample") const;

    void printStatsOf(Explanation const &, Network::Dataset const &, ExplanationIdx) const;
    void printStatsHeadOf(Network::Dataset const &, ExplanationIdx) const;
    void printStatsBodyOf(Explanation const &) const;

    Framework & framework;

    std::unique_ptr<xai::verifiers::Verifier> verifierPtr{};

    Strategies strategies{};

    bool requiresSMTSolver{false};

private:
    Network::Dataset::SampleIndices getSampleIndices(Network::Dataset const &) const;
};
} // namespace spexplain

#endif // SPEXPLAIN_EXPAND_H
