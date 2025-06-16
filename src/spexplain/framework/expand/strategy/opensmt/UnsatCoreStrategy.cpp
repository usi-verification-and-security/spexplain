#include "UnsatCoreStrategy.h"

#include <verifiers/opensmt/OpenSMTVerifier.h>

#include <api/MainSolver.h>

namespace spexplain::expand::opensmt {
void UnsatCoreStrategy::executeInit(Explanations & explanations, Network::Dataset const & data, ExplanationIdx idx) {
    Base::executeInit(explanations, data, idx);
    // if opensmt Strategy also has executeInit, only its local function should be called

    auto & verifier = getVerifier();
    auto & solver = verifier.getSolver();
    auto & solverConf = solver.getConfig();

    using ::opensmt::SMTConfig;
    using ::opensmt::SMTOption;

    char const * msg = "ok";
    solverConf.setOption(SMTConfig::o_produce_unsat_cores, SMTOption(true), msg);
    solverConf.setOption(SMTConfig::o_minimal_unsat_cores, SMTOption(config.minimal), msg);
}
} // namespace spexplain::expand::opensmt
