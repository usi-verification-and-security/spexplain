#ifndef SPEXPLAIN_CONJEXPLANATION_H
#define SPEXPLAIN_CONJEXPLANATION_H

#include "Explanation.h"

#include <memory>
#include <vector>

namespace spexplain {
class ConjunctExplanation : public Explanation {
public:
    using Conjunction = std::vector<std::unique_ptr<PartialExplanation>>;

    struct PrintConfig {
        char delim = ' ';
    };

    using Explanation::Explanation;
    explicit ConjunctExplanation(Framework const & fw, std::size_t size_) : Explanation{fw}, conjunction(size_) {}

    // May contain null pointers representing true values
    Conjunction::const_iterator begin() const { return conjunction.cbegin(); }
    Conjunction::const_iterator end() const { return conjunction.cend(); }
    Conjunction::iterator begin() { return conjunction.begin(); }
    Conjunction::iterator end() { return conjunction.end(); }

    // Including null pointers
    std::size_t size() const { return conjunction.size(); }
    // Discluding null ponters
    std::size_t validSize() const;

    bool isSparse() const { return validSize() < size(); }

    Conjunction::value_type const & operator[](std::size_t idx) const;
    Conjunction::value_type & operator[](std::size_t idx);

    // May return null pointer ~ true value
    PartialExplanation const * tryGetExplanation(std::size_t idx) const { return operator[](idx).get(); }
    PartialExplanation * tryGetExplanation(std::size_t idx) { return operator[](idx).get(); }

    bool contains(VarIdx) const override;

    std::size_t termSize() const override;

    void clear() override;

    void swap(ConjunctExplanation &);

    virtual void insertExplanation(std::unique_ptr<PartialExplanation>);
    // Effectively sets the partial explanation to true value
    bool eraseExplanation(std::size_t idx);
    bool eraseExplanation(Conjunction::iterator);

    virtual void condense();

    virtual void intersect(std::unique_ptr<Explanation> &&);
    virtual void intersect(ConjunctExplanation &&);

    void printSmtLib2(std::ostream &) const override;
    void printSmtLib2(std::ostream &, PrintConfig const &) const;

protected:
    virtual bool eraseExplanation(std::unique_ptr<PartialExplanation> &);

    //+ also store indices to a set and iterate using it if the vector is already too sparse
    // see `assert-interval-maybe-using-map` git tag
    Conjunction conjunction{};
};
} // namespace spexplain

#endif // SPEXPLAIN_CONJEXPLANATION_H
