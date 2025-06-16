#ifndef SPEXPLAIN_IVALEXPLANATION_H
#define SPEXPLAIN_IVALEXPLANATION_H

#include "ConjunctExplanation.h"

#include "VarBound.h"

#include <spexplain/common/Bound.h>
#include <spexplain/common/Interval.h>

#include <cassert>

namespace spexplain {
//! only some things are shared with ConjunctExplanation but not all - there should be a common base class
// then it will also hold that IntervalExplanation is not ConjunctExplanation, only e.g. ConjunctExplanationBase
class IntervalExplanation : public ConjunctExplanation {
public:
    enum class PrintFormat { smtlib2, bounds, intervals };

    struct PrintConfig : ConjunctExplanation::PrintConfig {
        bool includeAll;
    };

    static constexpr PrintConfig defaultBoundsPrintConfig{{.delim = '\n'}, false};
    static constexpr PrintConfig defaultIntervalsPrintConfig{{.delim = ' '}, true};

    explicit IntervalExplanation(Framework const &);

    bool supportsVolume() const override { return true; }

    std::size_t size() const {
        assert(ConjunctExplanation::size() == frameworkPtr->varSize());
        return ConjunctExplanation::size();
    }

    VarBound const * tryGetVarBound(VarIdx idx) const { return castToVarBound(tryGetExplanation(idx)); }
    VarBound * tryGetVarBound(VarIdx idx) { return castToVarBound(tryGetExplanation(idx)); }

    bool contains(VarIdx idx) const override { return bool(tryGetVarBound(idx)); }

    std::size_t varSize() const override;

    void insertVarBound(VarBound);
    void insertBound(VarIdx, Bound);

    bool eraseVarBound(VarIdx idx) { return eraseExplanation(idx); }

    void intersect(IntervalExplanation &&);

    std::unique_ptr<ConjunctExplanation> toConjunctExplanation(std::vector<VarIdx> const & varOrder) &&;

    Float getRelativeVolume() const override;
    Float getRelativeVolumeSkipFixed() const override;

    void print(std::ostream & os) const override;
    void printBounds(std::ostream & os) const { printBounds(os, defaultBoundsPrintConfig); }
    void printIntervals(std::ostream & os) const { printIntervals(os, defaultIntervalsPrintConfig); }
    void print(std::ostream &, PrintConfig const &) const;
    void printBounds(std::ostream &, PrintConfig const &) const;
    void printIntervals(std::ostream &, PrintConfig const &) const;

protected:
    std::size_t computeFixedCount() const override;

    PrintFormat const & getPrintFormat() const;

private:
    static VarBound const * castToVarBound(PartialExplanation const * expPtr) {
        assert(not expPtr or dynamic_cast<VarBound const *>(expPtr));
        return static_cast<VarBound const *>(expPtr);
    }

    static VarBound * castToVarBound(PartialExplanation * expPtr) {
        assert(not expPtr or dynamic_cast<VarBound *>(expPtr));
        return static_cast<VarBound *>(expPtr);
    }

    void insertExplanation(std::unique_ptr<PartialExplanation>) override;

    void condense() override;

    void intersect(std::unique_ptr<Explanation> &&) override;
    void intersect(ConjunctExplanation &&) override;

    template<bool skipFixed>
    Float computeRelativeVolumeTp() const;

    template<PrintFormat>
    void printTp(std::ostream &, PrintConfig const &) const;
    static void printElemBounds(std::ostream &, PrintConfig const &, VarBound const &);
    void printElemInterval(std::ostream &, PrintConfig const &, VarBound const &) const;
    void printElemBounds(std::ostream &, PrintConfig const &, VarIdx, VarBound const *) const;
    void printElemInterval(std::ostream &, PrintConfig const &, VarIdx, VarBound const *) const;
};
} // namespace spexplain

#endif // SPEXPLAIN_IVALEXPLANATION_H
