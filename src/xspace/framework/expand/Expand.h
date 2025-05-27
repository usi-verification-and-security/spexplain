#ifndef XSPACE_EXPAND_H
#define XSPACE_EXPAND_H

#include "../Framework.h"

#include <xspace/common/Var.h>
#include <xspace/nn/Dataset.h>

#include <memory>
#include <vector>

namespace xai::verifiers {
class Verifier;
}

namespace xspace {
class Explanation;

static_assert(std::is_same_v<ExplanationIdx, Dataset::Sample::Idx>);

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

    void operator()(Explanations &, Dataset const &);

protected:
    void addStrategy(std::unique_ptr<Strategy>);

    std::unique_ptr<xai::verifiers::Verifier> makeVerifier(std::string_view name) const;
    void setVerifier(std::unique_ptr<xai::verifiers::Verifier>);

    Dataset::SampleIndices makeSampleIndices(Dataset const &) const;

    void initVerifier();

    void assertModel();
    void resetModel();

    void assertClassification(Dataset::Output const &);
    void resetClassification();

    void printStatsHead(Dataset const &) const;
    void printStats(Explanation const &, Dataset const &, ExplanationIdx) const;

    Framework & framework;

    std::unique_ptr<xai::verifiers::Verifier> verifierPtr{};

    Strategies strategies{};

    bool requiresSMTSolver{false};

private:
    Dataset::SampleIndices getSampleIndices(Dataset const &) const;
};
} // namespace xspace

#endif // XSPACE_EXPAND_H
