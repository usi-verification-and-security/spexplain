#ifndef XAI_SMT_OPENSMTVERIFIER_H
#define XAI_SMT_OPENSMTVERIFIER_H

#include <verifiers/UnsatCoreVerifier.h>

#include <memory>

namespace opensmt {
class PTRef;
class MainSolver;
} // namespace opensmt

namespace xai::verifiers {

class OpenSMTVerifier : public UnsatCoreVerifier {
public:
    OpenSMTVerifier();
    virtual ~OpenSMTVerifier();
    OpenSMTVerifier(OpenSMTVerifier const &) = delete;
    OpenSMTVerifier & operator=(OpenSMTVerifier const &) = delete;
    OpenSMTVerifier(OpenSMTVerifier &&) = default;
    OpenSMTVerifier & operator=(OpenSMTVerifier &&) = default;

    bool contains(::opensmt::PTRef const &, NodeIndex) const;

    std::size_t termSizeOf(::opensmt::PTRef const &) const;

    void loadModel(spexplain::Network const &) override;

    void setUnsatCoreFilter(std::vector<NodeIndex> const &) override;

    void addTerm(::opensmt::PTRef const &);
    void addExplanationTerm(::opensmt::PTRef const &, std::string termNamePrefix = "");

    void addUpperBound(LayerIndex layer, NodeIndex var, float value, bool explanationTerm = false) override;
    void addLowerBound(LayerIndex layer, NodeIndex var, float value, bool explanationTerm = false) override;
    // Ensure that equalities and intervals correspond to just one assertion
    void addEquality(LayerIndex layer, NodeIndex var, float value, bool explanationTerm = false) override;
    void addInterval(LayerIndex layer, NodeIndex var, float lo, float hi, bool explanationTerm = false) override;

    void addClassificationConstraint(NodeIndex node, float threshold) override;

    void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, float rhs) override;

    void resetSampleQuery() override;
    void resetSample() override;
    void reset() override;

    UnsatCore getUnsatCore() const override;

    //+ remove from API
    opensmt::MainSolver const & getSolver() const;
    opensmt::MainSolver & getSolver();

    void printSmtLib2Query(std::ostream &) const override;

protected:
    void initImpl() override;

    void pushImpl() override;
    void popImpl() override;

    Answer checkImpl() override;

private:
    class OpenSMTImpl;
    std::unique_ptr<OpenSMTImpl> pimpl;
};

} // namespace xai::verifiers

#endif // XAI_SMT_OPENSMTVERIFIER_H
