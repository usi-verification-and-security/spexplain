#include "Print.h"

#include "Config.h"

#include <iostream>

namespace spexplain {
Framework::Print::Print(Framework const & fw) : framework{fw} {
    auto const & conf = framework.getConfig();

    if (not conf.isQuiet()) { infoOsPtr = &std::cout; }

    if (ignoringExplanations()) {
        assert(not conf.getExplanationsFileName().empty());
        setExplanationsFileName(conf.getExplanationsFileName());
    }
    if (ignoringStats() and conf.statsFileNameIsSet()) {
        assert(not conf.getStatsFileName().empty());
        setStatsFileName(conf.getStatsFileName());
    }
}

void Framework::Print::setExplanationsFileName(std::string_view fileName) {
    explanationsFileName = fileName;
    setExplanationsFile(fileName);
}

void Framework::Print::setExplanationsFile(std::string_view fileName) {
    explanationsFileOs = std::ofstream{std::string{fileName}};
    explanationsOsPtr = &explanationsFileOs;
}

void Framework::Print::setStatsFileName(std::string_view fileName) {
    statsFileName = fileName;
    setStatsFile(fileName);
}

void Framework::Print::setStatsFile(std::string_view fileName) {
    statsFileOs = std::ofstream{std::string{fileName}};
    statsOsPtr = &statsFileOs;
}
} // namespace spexplain
