#ifndef XSPACE_FRAMEWORK_PRINT_H
#define XSPACE_FRAMEWORK_PRINT_H

#include "Framework.h"

#include <xspace/common/Print.h>

#include <cassert>
#include <iosfwd>

namespace xspace {
class Framework::Print {
public:
    Print(Framework const &);

    bool ignoringExplanations() const { return ignoring(explanationsOsPtr); }
    bool ignoringStats() const { return ignoring(statsOsPtr); }

    std::ostream & explanations() const {
        assert(explanationsOsPtr);
        return *explanationsOsPtr;
    }
    std::ostream & stats() const {
        assert(statsOsPtr);
        return *statsOsPtr;
    }

protected:
    struct Absorb : std::ostream {
        std::ostream & operator<<(auto const &) { return *this; }
    };

    bool ignoring(std::ostream * osPtr) const {
        assert(osPtr);
        return osPtr == &absorb;
    }

    Framework const & framework;

    static inline Absorb absorb{};

    std::ostream * explanationsOsPtr{&absorb};
    std::ostream * statsOsPtr{&absorb};
};
} // namespace xspace

#endif // XSPACE_FRAMEWORK_PRINT_H
