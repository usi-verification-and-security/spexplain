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

    void assertSampleModel(spexplain::Network const &) override;

    void setUnsatCoreFilter(std::vector<NodeIndex> const &) override;

    void addTerm(::opensmt::PTRef const &);
    void addExplanationTerm(::opensmt::PTRef const &, std::string termNamePrefix = "");

    void addUpperBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;
    void addLowerBound(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;
    // Ensure that equalities and intervals correspond to just one assertion
    void addEquality(LayerIndex layer, NodeIndex var, Float value, bool explanationTerm = false) override;
    void addInterval(LayerIndex layer, NodeIndex var, Float lo, Float hi, bool explanationTerm = false) override;

    void addClassificationConstraint(NodeIndex node, Float threshold) override;

    void addConstraint(LayerIndex layer, std::vector<std::pair<NodeIndex, int>> lhs, Float rhs) override;

    void addPreference(::opensmt::PTRef const &);

    void setTimeLimit(std::chrono::milliseconds) override;

    void resetSampleQuery() override;
    void resetSample() override;
    void resetSampleModel() override;

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
