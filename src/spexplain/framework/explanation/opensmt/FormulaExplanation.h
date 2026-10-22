#ifndef SPEXPLAIN_OSMTEXPLANATION_H
#define SPEXPLAIN_OSMTEXPLANATION_H

#include "../Explanation.h"

#include <memory>

namespace opensmt {
struct PTRef;
} // namespace opensmt

namespace xai::verifiers {
class OpenSMTVerifier;
}

namespace spexplain::opensmt {
using Formula = ::opensmt::PTRef;

class FormulaExplanation : public Explanation {
public:
    using Explanation::Explanation;
    explicit FormulaExplanation(Framework const &, Formula const &);

    Formula const & getFormula() const { return *formulaPtr; }

    bool contains(VarIdx) const override;

    std::size_t termSize() const override;

    void clear() override;

    void swap(FormulaExplanation &);

    void intersect(FormulaExplanation &&);

    void printSmtLib2(std::ostream &) const override;

protected:
    xai::verifiers::OpenSMTVerifier const & getVerifier() const;

    void resetFormula();

    std::unique_ptr<Formula> formulaPtr{};
};
} // namespace spexplain::opensmt

#endif // SPEXPLAIN_OSMTEXPLANATION_H
