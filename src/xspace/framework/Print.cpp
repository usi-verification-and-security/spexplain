#include "Print.h"

#include "Config.h"

#include <iostream>

namespace xspace {
//+ move explanations into different output, probably a file
Framework::Print::Print(Framework const & fw) : framework{fw}, infoOsPtr{&std::cout}, explanationsOsPtr{&std::cout} {
    auto const & conf = framework.getConfig();
    bool const verbose = conf.isVerbose();

    if (not verbose) { return; }

    statsOsPtr = &std::cerr;
}
} // namespace xspace
