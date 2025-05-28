#ifndef XSPACE_PREPROCESS_H
#define XSPACE_PREPROCESS_H

#include "Framework.h"

#include <xspace/network/Network.h>

namespace xspace {
class Framework::Preprocess {
public:
    Preprocess(Framework &);

    void operator()(Network::Dataset &) const;

    Explanations makeExplanationsFromSamples(Network::Dataset const &) const;
    std::unique_ptr<Explanation> makeExplanationFromSample(Network::Sample const &) const;

protected:
    Framework & framework;
};
} // namespace xspace

#endif // XSPACE_PREPROCESS_H
