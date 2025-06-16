#ifndef SPEXPLAIN_PARTIALEXPLANATION_H
#define SPEXPLAIN_PARTIALEXPLANATION_H

#include "../Framework.h"

#include <spexplain/common/Var.h>

namespace spexplain {
class PartialExplanation {
public:
    explicit PartialExplanation(Framework const & fw) : frameworkPtr{&fw} {}
    virtual ~PartialExplanation() = default;
    PartialExplanation(PartialExplanation const &) = default;
    PartialExplanation(PartialExplanation &&) = default;
    PartialExplanation & operator=(PartialExplanation const &) = default;
    PartialExplanation & operator=(PartialExplanation &&) = default;

    virtual bool contains(VarIdx) const = 0;

    virtual std::size_t varSize() const;
    virtual std::size_t termSize() const = 0;

    void swap(PartialExplanation &);

    virtual void print(std::ostream & os) const { printSmtLib2(os); }
    virtual void printSmtLib2(std::ostream &) const = 0;

protected:
    Framework::Expand const & getExpand() const { return frameworkPtr->getExpand(); }

    Framework const * frameworkPtr;
};
} // namespace spexplain

#endif // SPEXPLAIN_PARTIALEXPLANATION_H
