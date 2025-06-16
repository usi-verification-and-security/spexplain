#ifndef SPEXPLAIN_BOUND_H
#define SPEXPLAIN_BOUND_H

#include "Core.h"

#include <cassert>
#include <iosfwd>
#include <type_traits>
#include <variant>

namespace spexplain {
class Bound {
public:
    enum class Type { eq, lteq, gteq };

    struct EqTag {};
    struct LtEqTag {};
    struct GtEqTag {};

    using LowerTag = GtEqTag;
    using UpperTag = LtEqTag;

    static constexpr Type lowerType = Type::gteq;
    static constexpr Type upperType = Type::lteq;

    Bound() = default;

    // Run-time type
    explicit Bound(Type const & type, Float val)
        : Bound(type == Type::eq     ? Bound(EqTag{}, val)
                : type == Type::lteq ? Bound(LtEqTag{}, val)
                                     : Bound(GtEqTag{}, val)) {}

    // Compile-time type
    explicit Bound(EqTag t, Float val) : Bound(t, val, OpConsTag{}) {}
    explicit Bound(LtEqTag t, Float val) : Bound(t, val, OpConsTag{}) {}
    explicit Bound(GtEqTag t, Float val) : Bound(t, val, OpConsTag{}) {}

    bool isEq() const { return std::holds_alternative<Eq>(op); }
    bool isLtEq() const { return std::holds_alternative<LtEq>(op); }
    bool isGtEq() const { return std::holds_alternative<GtEq>(op); }

    bool isLower() const { return isGtEq(); }
    bool isUpper() const { return isLtEq(); }

    Float getValue() const { return value; }

    char const * getSymbol() const {
        return std::visit([](auto & t) { return t.getSymbol(); }, op);
    }
    char const * getReverseSymbol() const {
        return std::visit([](auto & t) { return t.getReverseSymbol(); }, op);
    }

    void print(std::ostream & os) const { printRegular(os); }
    void printReverse(std::ostream &) const;

protected:
    struct Eq {
        char const * getSymbol() const { return "="; }
        char const * getReverseSymbol() const { return getSymbol(); }
    };
    struct LtEq {
        char const * getSymbol() const { return "<="; }
        char const * getReverseSymbol() const { return ">="; }
    };
    struct GtEq : LtEq {
        char const * getSymbol() const { return LtEq::getReverseSymbol(); }
        char const * getReverseSymbol() const { return LtEq::getSymbol(); }
    };

    template<typename T>
    using OpTp =
        std::conditional_t<std::is_same_v<T, EqTag>, Eq, std::conditional_t<std::is_same_v<T, LtEqTag>, LtEq, GtEq>>;

    using Op = std::variant<Eq, LtEq, GtEq>;

    struct OpConsTag {};

    template<typename T>
    explicit Bound(T, Float val, OpConsTag) : op{OpTp<T>{}},
                                              value{val} {}

    void printRegular(std::ostream &) const;

    Op op{};

    Float value{};
};

struct EqBound : public Bound {
    explicit EqBound(Float val) : Bound{EqTag{}, val} {}
    explicit EqBound(Bound const & bnd) : EqBound(bnd.getValue()) {}
};

struct LtEqBound : public Bound {
    explicit LtEqBound(Float val) : Bound{LtEqTag{}, val} {}
    explicit LtEqBound(Bound const & bnd) : LtEqBound(bnd.getValue()) {}
};

struct GtEqBound : public Bound {
    explicit GtEqBound(Float val) : Bound{GtEqTag{}, val} {}
    explicit GtEqBound(Bound const & bnd) : GtEqBound(bnd.getValue()) {}
};

using LowerBound = GtEqBound;
using UpperBound = LtEqBound;
} // namespace spexplain

#endif // SPEXPLAIN_BOUND_H
