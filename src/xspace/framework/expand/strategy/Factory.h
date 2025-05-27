#ifndef XSPACE_EXPAND_STRATEGY_FACTORY_H
#define XSPACE_EXPAND_STRATEGY_FACTORY_H

#include "Strategy.h"

#include <string>

namespace xspace {
class Framework::Expand::Strategy::Factory {
public:
    Factory(Expand &, VarOrdering const & = {});

    std::unique_ptr<Strategy> parse(std::string const &);

protected:
    template<typename T>
    std::unique_ptr<Strategy> parseReturnTp(std::string const &, auto const & params, auto &&...) const;

    template<typename T>
    [[noreturn]]
    void throwAdditionalParametersTp(std::string const &) const;
    template<typename T>
    [[noreturn]]
    void throwInvalidParameterTp(std::string const & param) const;

    template<typename T>
    std::string throwMessageTp(std::string const &) const;

    std::vector<VarIdx> parseVarIndices(std::istream &) const;

    Expand & expand;

    VarOrdering const & varOrdering;

private:
    template<typename T>
    std::unique_ptr<Strategy> newStrategyTp(auto &&...) const;

    template<typename T>
    void checkAdditionalParametersTp(std::string const &, auto const & params) const;

    template<typename StrategyT>
    std::unique_ptr<Strategy> parseDefault(std::string const &, auto & params);
    std::unique_ptr<Strategy> parseTrial(std::string const &, auto & params);
    std::unique_ptr<Strategy> parseUnsatCore(std::string const &, auto & params);
    std::unique_ptr<Strategy> parseInterpolation(std::string const &, auto & params);
    std::unique_ptr<Strategy> parseSlice(std::string const &, auto & params);
};
} // namespace xspace

#endif // XSPACE_EXPAND_STRATEGY_FACTORY_H
