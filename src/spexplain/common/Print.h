#ifndef SPEXPLAIN_PRINT_H
#define SPEXPLAIN_PRINT_H

#include "Core.h"

#include <iosfwd>

namespace spexplain {
template<typename T>
concept printable = requires(T const & t, std::ostream & os) { t.print(os); };

template<typename T>
concept printable_in_smtlib2 = requires(T const & t, std::ostream & os) { t.printSmtLib2(os); };

template<printable_in_smtlib2 T>
struct PrintSmtLib2Proxy {
    T const & t;

    friend std::ostream & operator<<(std::ostream & os, PrintSmtLib2Proxy const & rhs) {
        rhs.t.printSmtLib2(os);
        return os;
    }
};

inline std::ostream & operator<<(std::ostream & os, printable auto const & rhs) {
    rhs.print(os);
    return os;
}

template<printable_in_smtlib2 T>
inline auto smtLib2Format(T const & arg) {
    return PrintSmtLib2Proxy<T>{arg};
}

void printSmtLib2AsRational(std::ostream &, Float);
} // namespace spexplain

#endif // SPEXPLAIN_PRINT_H
