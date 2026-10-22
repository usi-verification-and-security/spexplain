#ifndef SPEXPLAIN_STRING_H
#define SPEXPLAIN_STRING_H

#include "Print.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace spexplain {
constexpr char const * whitespace = " \t\n\f\r\v";

inline std::string toString(printable auto const & arg) {
    std::ostringstream oss;
    arg.print(oss);
    return std::move(oss).str();
}

inline std::string_view toStringView(printable auto const & arg) {
    std::ostringstream oss;
    arg.print(oss);
    return std::move(oss).view();
}

inline std::string toLower(std::string_view sv) {
    std::string str{sv};
    std::ranges::transform(str, str.begin(), [](unsigned char c) { return std::tolower(c); });
    return str;
}

inline std::string toUpper(std::string_view sv) {
    std::string str{sv};
    std::ranges::transform(str, str.begin(), [](unsigned char c) { return std::toupper(c); });
    return str;
}

constexpr std::string_view ltrim(std::string_view sv) {
    size_t const pos = sv.find_first_not_of(whitespace);
    if (pos == std::string_view::npos) { return {}; }
    sv.remove_prefix(pos);
    return sv;
}

constexpr std::string_view rtrim(std::string_view sv) {
    size_t const pos = sv.find_last_not_of(whitespace);
    if (pos == std::string_view::npos) { return {}; }
    sv.remove_suffix(sv.size() - pos - 1);
    return sv;
}

constexpr std::string_view trim(std::string_view sv) {
    return rtrim(ltrim(sv));
}
} // namespace spexplain

#endif // SPEXPLAIN_STRING_H
