#ifndef SPEXPLAIN_FRAMEWORK_PRINT_H
#define SPEXPLAIN_FRAMEWORK_PRINT_H

#include "Framework.h"

#include <spexplain/common/Print.h>

#include <cassert>
#include <fstream>
#include <ostream>
#include <streambuf>

namespace spexplain {
class Framework::Print {
public:
    Print(Framework const &);

    void setExplanationsFileName(std::string_view fileName);
    void setStatsFileName(std::string_view fileName);
    void setTimesFileName(std::string_view fileName);

    bool ignoringInfo() const { return ignoring(infoOsPtr); }
    bool ignoringExplanations() const { return ignoring(explanationsOsPtr); }
    bool ignoringStats() const { return ignoring(statsOsPtr); }
    bool ignoringTimes() const { return ignoring(timesOsPtr); }

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
    std::ostream & times() const {
        assert(timesOsPtr);
        return *timesOsPtr;
    }

protected:
    Print(Print const &) = delete;
    Print & operator=(Print const &) = delete;

    /// Sink that discards everything written to it.
    /// It must own a real stream buffer: a default-constructed std::ostream leaves the
    /// basic_ios base uninitialized, so anything reaching the base operators -- `os << std::endl`,
    /// or any write through an `std::ostream &` parameter -- touches an uninitialized locale
    /// and segfaults. Hiding operator<< in this class is not enough, because it is only
    /// visible when the static type is Absorb, and the sink is always used as std::ostream &.
    struct Absorb : std::ostream {
        Absorb() : std::ostream{nullptr} { rdbuf(&buf); }

    private:
        struct NullBuf : std::streambuf {
            int_type overflow(int_type ch) override { return traits_type::not_eof(ch); }
        };

        NullBuf buf{};
    };

    void setExplanationsFile(std::string_view fileName);
    void setStatsFile(std::string_view fileName);
    void setTimesFile(std::string_view fileName);

    bool ignoring(std::ostream * osPtr) const {
        assert(osPtr);
        return osPtr == &absorb;
    }

    Framework const & framework;

    static inline Absorb absorb{};

    std::ostream * infoOsPtr{&absorb};
    std::ostream * explanationsOsPtr{&absorb};
    std::ostream * statsOsPtr{&absorb};
    std::ostream * timesOsPtr{&absorb};

    std::string explanationsFileName{};
    std::string statsFileName{};
    std::string timesFileName{};
    std::ofstream explanationsFileOs{};
    std::ofstream statsFileOs{};
    std::ofstream timesFileOs{};
};
} // namespace spexplain

#endif // SPEXPLAIN_FRAMEWORK_PRINT_H
