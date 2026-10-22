#ifndef XAI_SMT_UCOREVERIFIER_H
#define XAI_SMT_UCOREVERIFIER_H

#include "Verifier.h"

#include <vector>

namespace xai::verifiers {

struct UnsatCore {
    // Based on the order of the assertions
    // Beware, equalities and intervals may in general produce 2 assertions
    std::vector<std::size_t> includedIndices;
    std::vector<std::size_t> excludedIndices;

    std::vector<NodeIndex> lowerBounds;
    std::vector<NodeIndex> upperBounds;
    std::vector<NodeIndex> equalities;
    std::vector<NodeIndex> intervals;
};

class UnsatCoreVerifier : public Verifier {
public:
    virtual void setUnsatCoreFilter(std::vector<NodeIndex> const &) = 0;

    virtual UnsatCore getUnsatCore() const = 0;
};
} // namespace xai::verifiers

#endif // XAI_SMT_UCOREVERIFIER_H
