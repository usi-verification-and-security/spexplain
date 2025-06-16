#ifndef XAI_SMT_VERIFIER_H
#define XAI_SMT_VERIFIER_H

#include <spexplain/network/Network.h>

#include <string>
#include <vector>

//++ move into spexplain namespace
namespace xai::verifiers {

using NodeIndex = std::size_t;
using LayerIndex = std::size_t;

class Verifier {
public:
    enum class Answer { SAT, UNSAT, UNKNOWN, ERROR };

    Verifier() = default;
    virtual ~Verifier() = default;
    Verifier(Verifier const &) = delete;
    Verifier & operator=(Verifier const &) = delete;
    Verifier(Verifier &&) = default;
    Verifier & operator=(Verifier &&) = default;

    virtual void loadModel(spexplain::Network const &) = 0;

    virtual void addUpperBound(LayerIndex layer, NodeIndex var, float value, bool explanationTerm = false) = 0;
    virtual void addLowerBound(LayerIndex layer, NodeIndex var, float value, bool explanationTerm = false) = 0;
    virtual void addEquality(LayerIndex layer, NodeIndex var, float value, bool explanationTerm = false) {
        addInterval(layer, var, value, value, explanationTerm);
    }
    virtual void addInterval(LayerIndex layer, NodeIndex var, float lo, float hi, bool explanationTerm = false) {
        addUpperBound(layer, var, hi, explanationTerm);
        addLowerBound(layer, var, lo, explanationTerm);
    }

    virtual void addClassificationConstraint(NodeIndex node, float threshold) = 0;

    virtual void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, float rhs) = 0;

    virtual void init() {
        initImpl();
        reset();
    }

    virtual void push() { pushImpl(); }
    virtual void pop() { popImpl(); }

    virtual Answer check() {
        ++checksCount;
        return checkImpl();
    }

    std::size_t getChecksCount() const { return checksCount; }

    virtual void resetSampleQuery() {}
    virtual void resetSample() {
        resetSampleQuery();
        checksCount = 0;
    }
    virtual void reset() { resetSample(); }

    virtual void printSmtLib2Query(std::ostream &) const = 0;

protected:
    virtual void initImpl() {}

    std::size_t checksCount{};

private:
    virtual void pushImpl() = 0;
    virtual void popImpl() = 0;

    virtual Answer checkImpl() = 0;
};
} // namespace xai::verifiers

#endif // XAI_SMT_VERIFIER_H
