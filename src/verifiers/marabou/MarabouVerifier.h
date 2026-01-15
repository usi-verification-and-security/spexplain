#ifndef XAI_SMT_MARABOUVERIFIER_H
#define XAI_SMT_MARABOUVERIFIER_H

#include <verifiers/Verifier.h>

#include <memory>

namespace xai::verifiers {

class MarabouVerifier : public Verifier {
public:
    MarabouVerifier();
    virtual ~MarabouVerifier();
    MarabouVerifier(MarabouVerifier const &) = delete;
    MarabouVerifier & operator=(MarabouVerifier const &) = delete;
    MarabouVerifier(MarabouVerifier &&) = default;
    MarabouVerifier & operator=(MarabouVerifier &&) = default;

    void loadModel(spexplain::Network const &) override;

    void addUpperBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;

    void addLowerBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;

    void addClassificationConstraint(NodeIndex node, Float threshold) override;

    void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, Float rhs) override;

    void printSmtLib2Query(std::ostream &) const override;

protected:
    void pushImpl() override;
    void popImpl() override;

    Answer checkImpl() override;

private:
    class MarabouImpl;
    std::unique_ptr<MarabouImpl> pimpl;
};
} // namespace xai::verifiers

#endif // XAI_SMT_MARABOUVERIFIER_H
