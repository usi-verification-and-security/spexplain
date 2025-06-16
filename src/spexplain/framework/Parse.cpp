#include "Parse.h"

#include "explanation/IntervalExplanation.h"
#include "explanation/VarBound.h"

#include <spexplain/common/Macro.h>
#include <spexplain/common/String.h>
#include <spexplain/network/Dataset.h>

#include <cassert>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace spexplain {
Framework::Parse::Parse(Framework & fw) : framework{fw} {
    assert(not framework.varNames.empty());
}

Explanations Framework::Parse::parseIntervalExplanations(std::string_view fileName, Network::Dataset const & data) const {
    std::ifstream ifs{std::string{fileName}};
    if (not ifs.good()) { throw std::ifstream::failure{"Could not open explanations file "s + std::string{fileName}}; }

    //+ not assuming other formats so far
    return parseIntervalExplanationsSmtLib2(ifs, data);
}

Explanations Framework::Parse::parseIntervalExplanationsSmtLib2(std::istream & is, Network::Dataset const & data) const {
    std::size_t const maxSize = data.size();
    Explanations explanations;
    explanations.reserve(maxSize);

    std::string line;
    while (std::getline(is, line)) {
        std::istringstream iss{std::move(line)};
        if (auto explanationPtr = parseIntervalExplanationSmtLib2(iss)) {
            explanations.push_back(std::move(explanationPtr));
            continue;
        }

        throw std::logic_error{"Invalid interval explanation:\n"s + std::move(iss).str()};
    }

    assert(explanations.size() <= maxSize);
    return explanations;
}

namespace {
    //+ only assertions
    long parseInt(std::istream & is) {
        assert(is);
        long num;
        is >> num;
        assert(is);
        return num;
    }

    Float parseFloat(std::istream & is) {
        char c;

        assert(is);
        is >> c;
        assert(is);
        if (std::isdigit(c) or c == '-') {
            is.putback(c);
            return parseInt(is);
        }

        assert(c == '(');
        is >> c;
        assert(is and c == '/');
        auto num = parseInt(is);
        auto den = parseInt(is);
        is >> c;
        assert(is and c == ')');
        return static_cast<double>(num) / den;
    }

    std::pair<VarIdx, Bound> parseBound(std::istream & is, std::string & str) {
        char c;

        Bound::Type type;
        if (str == "<=") {
            type = Bound::Type::lteq;
        } else if (str == ">=") {
            type = Bound::Type::gteq;
        } else if (str == "=") {
            type = Bound::Type::eq;
        } else {
            return {invalidVarIdx, {}};
        }

        is >> str;
        if (not is or str[0] != 'x') { return {invalidVarIdx, {}}; }
        VarIdx varIdx = getVarIdx(str);
        Float val = parseFloat(is);
        is >> c;
        if (not is or c != ')') { return {invalidVarIdx, {}}; }
        return {varIdx, Bound{type, val}};
    }
} // namespace

std::unique_ptr<Explanation> Framework::Parse::parseIntervalExplanationSmtLib2(std::istream & is) const {
    char c;
    std::string str;

    IntervalExplanation iexplanation{framework};

    is >> c >> str;
    if (not is or c != '(' or str != "and") { return nullptr; }
    while (is >> c and c != ')') {
        if (c != '(') { return nullptr; }
        is >> str;
        if (not is) { return nullptr; }
        if (str != "and") {
            auto [varIdx, bnd] = parseBound(is, str);
            if (varIdx == invalidVarIdx) { return nullptr; }
            iexplanation.insertBound(varIdx, std::move(bnd));
            continue;
        }

        is >> c >> str;
        if (not is or c != '(') { return nullptr; }
        auto [varIdx1, lo] = parseBound(is, str);
        is >> c >> str;
        if (not is or c != '(') { return nullptr; }
        auto [varIdx2, hi] = parseBound(is, str);
        if (varIdx1 == invalidVarIdx or varIdx1 != varIdx2) { return nullptr; }
        if (not lo.isLower() or not hi.isUpper()) { return nullptr; }
        is >> c;
        if (not is or c != ')') { return nullptr; }
        iexplanation.insertVarBound(VarBound{framework, varIdx1, std::move(lo), std::move(hi)});
    }

    return MAKE_UNIQUE(std::move(iexplanation));
}
} // namespace spexplain
