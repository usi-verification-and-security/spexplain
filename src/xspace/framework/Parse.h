#ifndef XSPACE_PARSE_H
#define XSPACE_PARSE_H

#include "Framework.h"

#include <xspace/network/Network.h>

namespace xspace {
class Framework::Parse {
public:
    Parse(Framework &);

    Explanations parseIntervalExplanations(std::string_view fileName, Network::Dataset const &) const;

protected:
    Explanations parseIntervalExplanationsSmtLib2(std::istream &, Network::Dataset const &) const;
    std::unique_ptr<Explanation> parseIntervalExplanationSmtLib2(std::istream &) const;

    Framework & framework;
};
} // namespace xspace

#endif // XSPACE_PARSE_H
