#ifndef XAI_SMT_VERIFIER_H
#define XAI_SMT_VERIFIER_H

#include <spexplain/network/Containers.h>
#include <spexplain/network/Network.h>

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

namespace spexplain {
class Explanation;
class Framework;
} // namespace spexplain

//++ move into spexplain namespace
namespace xai::verifiers {

using NodeIndex = std::size_t;
using LayerIndex = std::size_t;

using spexplain::Float;

class Verifier {
public:
    enum class Answer { SAT, UNSAT, UNKNOWN, ERROR };

    Verifier() = default;
    virtual ~Verifier() = default;
    Verifier(Verifier const &) = delete;
    Verifier & operator=(Verifier const &) = delete;
    Verifier(Verifier &&) = default;
    Verifier & operator=(Verifier &&) = default;

    void fixNeuronActivation(LayerIndex, NodeIndex, bool activation);
    void preferNeuronActivation(LayerIndex, NodeIndex, bool activation);
    // Do not insert neuron activation if one already exists
    bool tryFixNeuronActivation(LayerIndex, NodeIndex, bool activation);
    bool tryPreferNeuronActivation(LayerIndex, NodeIndex, bool activation);

    std::optional<bool> getFixedNeuronActivation(LayerIndex, NodeIndex) const;
    std::optional<bool> getPreferredNeuronActivation(LayerIndex, NodeIndex) const;

    virtual void assertGroundModel() {}
    virtual void assertSampleModel() = 0;

    virtual void addUpperBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) = 0;
    virtual void addLowerBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) = 0;
    virtual void addEquality(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) {
        addInterval(layer, var, value, value, explanationTerm);
    }
    virtual void addInterval(LayerIndex layer, NodeIndex var, Float lo, Float hi, bool explanationTerm = false) {
        addUpperBound(layer, var, hi, explanationTerm);
        addLowerBound(layer, var, lo, explanationTerm);
    }

    virtual void addClassificationConstraint(NodeIndex node, Float threshold) = 0;

    virtual void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, Float rhs) = 0;

    virtual void init(spexplain::Network const &);

    virtual void push() { pushImpl(); }
    virtual void pop() { popImpl(); }

    virtual void setTimeLimit(std::chrono::milliseconds) {}

    virtual Answer check() {
        ++checksCount;
        return checkImpl();
    }

    std::size_t getChecksCount() const { return checksCount; }

    virtual std::unique_ptr<spexplain::Explanation> getSampleModelRestrictions(spexplain::Framework const &);

    virtual void resetSampleQuery() {}
    virtual void resetSample();
    virtual void resetSampleModel() { resetSample(); }
    virtual void resetGroundModel() {}

    virtual void reset() {
        resetSampleModel();
        resetGroundModel();
    }

    virtual void printSmtLib2Query(std::ostream &) const = 0;

protected:
    spexplain::Network const & getNetwork() const;

    virtual void initImpl(spexplain::Network const &);

    std::size_t checksCount{};

private:
    virtual void pushImpl() = 0;
    virtual void popImpl() = 0;

    virtual Answer checkImpl() = 0;

    spexplain::Network const * networkPtr;

    spexplain::NetworkMap<bool> fixedNeuronActivations;
    spexplain::NetworkMap<bool> preferredNeuronActivations;
};
} // namespace xai::verifiers

#endif // XAI_SMT_VERIFIER_H
