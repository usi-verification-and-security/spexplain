#ifndef XSPACE_EXPAND_H
#define XSPACE_EXPAND_H

#include "../Framework.h"

#include <xspace/common/Var.h>
#include <xspace/network/Dataset.h>

#include <memory>
#include <vector>

namespace xai::verifiers {
class Verifier;
}

namespace xspace {
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

    Expand(Framework &);

    Framework const & getFramework() const { return framework; }

    void setStrategies(std::istream &);

    void setVerifier();
    void setVerifier(std::string_view name);

    xai::verifiers::Verifier const & getVerifier() const {
        assert(verifierPtr);
        return *verifierPtr;
    }

    void dumpClassificationsAsSmtLib2Queries();
    void printClassificationAsSmtLib2Query(std::ostream &, Network::Classification const &);

    void operator()(Explanations &, Network::Dataset const &);

protected:
    void addStrategy(std::unique_ptr<Strategy>);

    std::unique_ptr<xai::verifiers::Verifier> makeVerifier(std::string_view name) const;
    void setVerifier(std::unique_ptr<xai::verifiers::Verifier>);

    Network::Dataset::SampleIndices makeSampleIndices(Network::Dataset const &) const;

    void initVerifier();

    void assertModel();
    void resetModel();

    void assertClassification(Network::Classification const &);
    void resetClassification();

    void printStatsHead(Network::Dataset const &) const;
    void printStats(Explanation const &, Network::Dataset const &, ExplanationIdx) const;

    Framework & framework;

    std::unique_ptr<xai::verifiers::Verifier> verifierPtr{};

    Strategies strategies{};

    bool requiresSMTSolver{false};

private:
    Network::Dataset::SampleIndices getSampleIndices(Network::Dataset const &) const;
};
} // namespace xspace

#endif // XSPACE_EXPAND_H
