#ifndef SPEXPLAIN_PARSE_H
#define SPEXPLAIN_PARSE_H

#include "Framework.h"

#include <spexplain/network/Network.h>

namespace spexplain {
class Framework::Parse {
public:
    Parse(Framework &);

    Explanations parseIntervalExplanations(std::string_view fileName, Network::Dataset const &) const;

protected:
    Explanations parseIntervalExplanationsSmtLib2(std::istream &, Network::Dataset const &) const;
    std::unique_ptr<Explanation> parseIntervalExplanationSmtLib2(std::istream &) const;

    Framework & framework;
};
} // namespace spexplain

#endif // SPEXPLAIN_PARSE_H
