#include "Print.h"

#include "String.h"

//+ ideally get rid of these
#include <common/StringConv.h>
#include <common/numbers/FastRational.h>

#include <ostream>
#include <string>

namespace spexplain {
namespace {
    //+ to be consistent with BasicVerix, but ideally should not depend on OpenSMT
    opensmt::FastRational floatToFastRational(Float value) {
        auto s = floatToDecimalString(value);
        char * rationalString;
        opensmt::stringToRational(rationalString, s.c_str());
        auto res = opensmt::FastRational(rationalString);
        free(rationalString);
        return res;
    }
} // namespace

void printSmtLib2AsRational(std::ostream & os, Float val) {
    auto const rat = floatToFastRational(val);
    // FastRational prints the large non-integer rationals as `n/d`, not in SMT-LIB, unlike integers
    auto const num = rat.get_num();
    auto const den = rat.get_den();
    if (den == 1) {
        os << num;
        return;
    }
    os << "(/ " << num << ' ' << den << ')';
}
} // namespace spexplain
