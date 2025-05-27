#include "Factory.h"

#include "Strategies.h"

#include <xspace/common/Macro.h>
#include <xspace/common/String.h>

#include <queue>
#include <sstream>
#include <stdexcept>

namespace xspace {
Framework::Expand::Strategy::Factory::Factory(Expand & exp, VarOrdering const & order)
    : expand{exp},
      varOrdering{std::move(order)} {}

std::unique_ptr<Framework::Expand::Strategy> Framework::Expand::Strategy::Factory::parse(std::string const & str) {
    static constexpr char paramDelim = ',';

    std::istringstream iss{str};
    std::string name;
    iss >> name;
    std::queue<std::string> params;
    std::string param;
    while (std::getline(iss, param, paramDelim)) {
        param = trim(param);
        params.push(std::move(param));
    }

    auto const nameLower = toLower(name);

    if (nameLower == NopStrategy::name()) { return parseDefault<NopStrategy>(str, params); }
    if (nameLower == AbductiveStrategy::name()) { return parseDefault<AbductiveStrategy>(str, params); }
    if (nameLower == TrialAndErrorStrategy::name()) { return parseTrial(str, params); }
    if (nameLower == expand::opensmt::UnsatCoreStrategy::name()) { return parseUnsatCore(str, params); }
    if (nameLower == expand::opensmt::InterpolationStrategy::name()) { return parseInterpolation(str, params); }
    if (nameLower == SliceStrategy::name()) { return parseSlice(str, params); }

    throw std::invalid_argument{"Unrecognized strategy name: "s + name};
}

template<typename T>
std::unique_ptr<Framework::Expand::Strategy>
Framework::Expand::Strategy::Factory::parseReturnTp(std::string const & str, auto const & params,
                                                    auto &&... args) const {
    checkAdditionalParametersTp<T>(str, params);
    return newStrategyTp<T>(FORWARD(args)...);
}

template<typename T>
std::unique_ptr<Framework::Expand::Strategy>
Framework::Expand::Strategy::Factory::newStrategyTp(auto &&... args) const {
    return std::make_unique<T>(expand, FORWARD(args)..., varOrdering);
}

template<typename T>
void Framework::Expand::Strategy::Factory::checkAdditionalParametersTp(std::string const & str,
                                                                       auto const & params) const {
    if (params.empty()) { return; }
    throwAdditionalParametersTp<T>(str);
};

template<typename T>
void Framework::Expand::Strategy::Factory::throwAdditionalParametersTp(std::string const & str) const {
    throw std::invalid_argument{throwMessageTp<T>("Additional parameters: "s + str)};
}

template<typename T>
void Framework::Expand::Strategy::Factory::throwInvalidParameterTp(std::string const & param) const {
    throw std::invalid_argument{throwMessageTp<T>("Invalid parameter: "s + param)};
}

template<typename T>
std::string Framework::Expand::Strategy::Factory::throwMessageTp(std::string const & msg) const {
    return "Strategy "s + T::name() + ": " + msg;
}

std::vector<VarIdx> Framework::Expand::Strategy::Factory::parseVarIndices(std::istream & is) const {
    std::vector<VarIdx> varIndices;

    //+ not handled when no vars are provided
    std::string param;
    while (is >> param) {
        //+ no checks
        VarName const & varName = param;
        VarIdx idx = getVarIdx(varName);
        varIndices.push_back(idx);
    }

    return varIndices;
}

template<typename StrategyT>
std::unique_ptr<Framework::Expand::Strategy> Framework::Expand::Strategy::Factory::parseDefault(std::string const & str,
                                                                                                auto & params) {
    return parseReturnTp<StrategyT>(str, params);
}

std::unique_ptr<Framework::Expand::Strategy> Framework::Expand::Strategy::Factory::parseTrial(std::string const & str,
                                                                                              auto & params) {
    TrialAndErrorStrategy::Config conf;
    while (not params.empty()) {
        std::string const paramStr = std::move(params.front());
        std::istringstream iss{paramStr};
        params.pop();
        std::string param;
        if (iss >> param) {
            auto const paramLower = toLower(param);
            if (paramLower == "n") {
                if (iss >> conf.maxAttempts) { continue; }
            }
        }

        throwInvalidParameterTp<TrialAndErrorStrategy>(paramStr);
    }

    return parseReturnTp<TrialAndErrorStrategy>(str, params, conf);
}

std::unique_ptr<Framework::Expand::Strategy>
Framework::Expand::Strategy::Factory::parseUnsatCore(std::string const & str, auto & params) {
    using expand::opensmt::UnsatCoreStrategy;

    UnsatCoreStrategy::Base::Config baseConf;
    UnsatCoreStrategy::Config conf;
    while (not params.empty()) {
        auto param = std::move(params.front());
        params.pop();
        auto const paramLower = toLower(param);
        if (paramLower == "sample") {
            baseConf.splitIntervals = false;
            continue;
        }

        if (paramLower == "interval") {
            baseConf.splitIntervals = true;
            continue;
        }

        if (paramLower == "min") {
            conf.minimal = true;
            continue;
        }

        throwInvalidParameterTp<UnsatCoreStrategy>(param);
    }

    return parseReturnTp<UnsatCoreStrategy>(str, params, baseConf, conf);
}

std::unique_ptr<Framework::Expand::Strategy>
Framework::Expand::Strategy::Factory::parseInterpolation(std::string const & str, auto & params) {
    using expand::opensmt::InterpolationStrategy;

    InterpolationStrategy::Config conf;
    while (not params.empty()) {
        std::string const paramStr = std::move(params.front());
        std::istringstream iss{paramStr};
        params.pop();
        std::string param;
        iss >> param;
        if (not iss) { throwInvalidParameterTp<InterpolationStrategy>(paramStr); }
        auto const paramLower = toLower(param);

        if (paramLower == "weak") {
            conf.boolInterpolationAlg = InterpolationStrategy::BoolInterpolationAlg::weak;
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::weak;
            continue;
        }

        if (paramLower == "strong") {
            conf.boolInterpolationAlg = InterpolationStrategy::BoolInterpolationAlg::strong;
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::strong;
            continue;
        }

        if (paramLower == "weaker") {
            conf.boolInterpolationAlg = InterpolationStrategy::BoolInterpolationAlg::weak;
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::weaker;
            continue;
        }

        if (paramLower == "stronger") {
            conf.boolInterpolationAlg = InterpolationStrategy::BoolInterpolationAlg::strong;
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::stronger;
            continue;
        }

        if (paramLower == "bweak") {
            conf.boolInterpolationAlg = InterpolationStrategy::BoolInterpolationAlg::weak;
            continue;
        }

        if (paramLower == "bstrong") {
            conf.boolInterpolationAlg = InterpolationStrategy::BoolInterpolationAlg::strong;
            continue;
        }

        if (paramLower == "aweak") {
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::weak;
            continue;
        }

        if (paramLower == "astrong") {
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::strong;
            continue;
        }

        if (paramLower == "aweaker") {
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::weaker;
            continue;
        }

        if (paramLower == "astronger") {
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::stronger;
            continue;
        }

        if (paramLower == "afactor") {
            conf.arithInterpolationAlg = InterpolationStrategy::ArithInterpolationAlg::factor;

            float factor;
            iss >> factor;
            if (not iss) { throwInvalidParameterTp<InterpolationStrategy>(paramStr); }
            //+ additional args not handled
            conf.arithInterpolationAlgFactor = factor;
            continue;
        }

        if (paramLower == "vars") {
            conf.varIndicesFilter = parseVarIndices(iss);
            continue;
        }

        throwInvalidParameterTp<InterpolationStrategy>(paramStr);
    }

    return parseReturnTp<InterpolationStrategy>(str, params, conf);
}

std::unique_ptr<Framework::Expand::Strategy> Framework::Expand::Strategy::Factory::parseSlice(std::string const & str,
                                                                                              auto & params) {
    SliceStrategy::Config conf;

    std::string const paramStr = std::move(params.front());
    std::istringstream iss{paramStr};
    params.pop();

    conf.varIndices = parseVarIndices(iss);

    return parseReturnTp<SliceStrategy>(str, params, conf);
}
} // namespace xspace
