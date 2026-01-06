#ifndef SPEXPLAIN_PREPROCESS_H
#define SPEXPLAIN_PREPROCESS_H

#include "Framework.h"

#include <spexplain/network/Network.h>

namespace spexplain {
class Framework::Preprocess {
public:
    Preprocess(Framework &);

    void operator()(Network::Dataset &) const;

    Explanations makeExplanationsFromSamples(Network::Dataset const &) const;
    std::unique_ptr<Explanation> makeExplanationFromSample(Sample const &) const;

protected:
    Framework & framework;
};
} // namespace spexplain

#endif // SPEXPLAIN_PREPROCESS_H
