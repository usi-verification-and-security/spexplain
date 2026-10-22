#ifndef SPEXPLAIN_EXPLANATION_H
#define SPEXPLAIN_EXPLANATION_H

#include "PartialExplanation.h"

#include <spexplain/common/Core.h>

namespace spexplain {
class Explanation : public PartialExplanation {
public:
    using PartialExplanation::PartialExplanation;

    virtual bool supportsVolume() const { return false; }

    virtual void clear() {}

    void swap(Explanation &);

    std::size_t getFixedCount() const { return computeFixedCount(); }

    virtual Float getRelativeVolume() const { return -1; }
    virtual Float getRelativeVolumeSkipFixed() const { return -1; }

protected:
    //! optimistic
    virtual std::size_t computeFixedCount() const { return 0; }
};
} // namespace spexplain

#endif // SPEXPLAIN_EXPLANATION_H
