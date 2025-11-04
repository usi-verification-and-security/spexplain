#ifndef SPEXPLAIN_FRAMEWORK_PRINT_H
#define SPEXPLAIN_FRAMEWORK_PRINT_H

#include "Framework.h"

#include <spexplain/common/Print.h>

#include <cassert>
#include <fstream>
#include <ostream>

namespace spexplain {
class Framework::Print {
public:
    Print(Framework const &);

    void setExplanationsFileName(std::string_view fileName);
    void setStatsFileName(std::string_view fileName);

    bool ignoringInfo() const { return ignoring(infoOsPtr); }
    bool ignoringExplanations() const { return ignoring(explanationsOsPtr); }
    bool ignoringStats() const { return ignoring(statsOsPtr); }

    std::ostream & info() const {
        assert(infoOsPtr);
        return *infoOsPtr;
    }
    std::ostream & explanations() const {
        assert(explanationsOsPtr);
        return *explanationsOsPtr;
    }
    std::ostream & stats() const {
        assert(statsOsPtr);
        return *statsOsPtr;
    }

protected:
    Print(Print const &) = delete;
    Print & operator=(Print const &) = delete;

    struct Absorb : std::ostream {
        std::ostream & operator<<(auto const &) { return *this; }
    };

    void setExplanationsFile(std::string_view fileName);
    void setStatsFile(std::string_view fileName);

    bool ignoring(std::ostream * osPtr) const {
        assert(osPtr);
        return osPtr == &absorb;
    }

    Framework const & framework;

    static inline Absorb absorb{};

    std::ostream * infoOsPtr{&absorb};
    std::ostream * explanationsOsPtr{&absorb};
    std::ostream * statsOsPtr{&absorb};

    std::string explanationsFileName{};
    std::string statsFileName{};
    std::ofstream explanationsFileOs{};
    std::ofstream statsFileOs{};
};
} // namespace spexplain

#endif // SPEXPLAIN_FRAMEWORK_PRINT_H
