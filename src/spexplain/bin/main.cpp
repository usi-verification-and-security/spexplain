#include <spexplain/framework/Config.h>
#include <spexplain/framework/Framework.h>
#include <spexplain/framework/expand/strategy/Strategies.h>
#include <spexplain/framework/explanation/Explanation.h>
#include <spexplain/network/Dataset.h>
#include <spexplain/network/Network.h>

#include <iomanip>
#include <iostream>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <getopt.h>

namespace {
void printUsageStrategyRow(std::ostream & os, std::string_view name, std::vector<std::string_view> params = {}) {
    os << std::setw(12) << name << ":";
    if (not params.empty()) {
        os << " " << params.front();
        for (auto param : params | std::views::drop(1)) {
            os << ", " << param;
        }
    }
    os << '\n';
}

void printUsageOptRow(std::ostream & os, char opt, std::string_view arg, std::string_view desc) {
    os << "    -" << opt << ' ';
    os << std::left << std::setw(7);
    if (arg.empty()) {
        os << ' ';
    } else {
        os << arg;
    }
    os << desc << '\n';
}

void printUsage(char * const argv[], std::ostream & os = std::cout) {
    using spexplain::Framework;
    using spexplain::expand::opensmt::InterpolationStrategy;
    using spexplain::expand::opensmt::UnsatCoreStrategy;

    assert(argv);
    std::string const cmd = argv[0];

    os << "USAGE: " << cmd;
    os << " [<action>] <args> [<options>]\n";

    os << "ACTIONS: [explain] dump-psi\n";
    os << "ARGS:\n";
    os << "\t explain:\t <nn_model_fn> <dataset_fn> <exp_strategies_spec>\n";
    os << "\t dump-psi:\t <nn_model_fn>\n";

    os << "STRATEGIES SPEC: '<spec1>[; <spec2>]...'\n";
    os << "Each spec: '<name>[ <param>[, <param>]...]'\n";
    os << "Strategies and parameters:\n";
    //+ template by the strategy and move the params to the classes as well
    printUsageStrategyRow(os, Framework::Expand::NopStrategy::name());
    printUsageStrategyRow(os, Framework::Expand::AbductiveStrategy::name());
    //+ also include 'vars'
    printUsageStrategyRow(os, Framework::Expand::TrialAndErrorStrategy::name(), {"n <int>"});
    printUsageStrategyRow(os, UnsatCoreStrategy::name(), {"sample", "interval", "min", "vars x<i>..."});
    printUsageStrategyRow(os, InterpolationStrategy::name(),
                          {"weak", "strong", "weaker", "stronger", "bweak", "bstrong", "aweak", "astrong", "aweaker",
                           "astronger", "afactor <factor>", "vars x<i>..."});
    printUsageStrategyRow(os, Framework::Expand::SliceStrategy::name(), {"[vars] x<i>..."});

    os << "VERIFIERS: opensmt";
#ifdef MARABOU
    os << " marabou";
#endif
    os << '\n';

    os << "OPTIONS:\n";
    printUsageOptRow(os, 'h', "", "Prints this help message and exits");
    printUsageOptRow(os, 'V', "<name>", "Set the verifier");
    printUsageOptRow(os, 'E', "<file>", "Use explanations from the file as starting points");
    printUsageOptRow(os, 'e', "<file>",
                     "Output explanations into the file (default: "s + Framework::Config::defaultExplanationsFileName +
                         ")");
    printUsageOptRow(os, 's', "<file>", "Output statistics into the file");
    printUsageOptRow(os, 'v', "", "Run in verbose mode");
    printUsageOptRow(os, 'q', "", "Run in quiet mode");
    printUsageOptRow(os, 'R', "", "Reverse the order of variables");
    printUsageOptRow(os, 'S', "", "Print the resulting explanations in the SMT-LIB2 format");
    printUsageOptRow(os, 'I', "", "Print the resulting explanations in the form of intervals");
    printUsageOptRow(os, 'r', "", "Shuffle (randomize) samples");
    printUsageOptRow(os, 'n', "<int>", "Maximum no. samples to be processed");

    os << "\nEXAMPLES:\n";
    os << cmd << " explain data/models/toy.nnet data/datasets/toy.csv abductive -e data/explanations/toy.phi.txt\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'ucore interval, min' -RvS -e toy.phi.txt\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'itp aweaker, bstrong; ucore'\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'trial n 2' -n1 -s stats.txt\n";
    os << cmd << " dump-psi data/models/toy.nnet\n";

    os.flush();
}

std::optional<int> getOpts(int argc, char * argv[], spexplain::Framework::Config & config,
                           std::string * explanationsFnPtr = nullptr) {

    int selectedLongOpt = 0;
    // constexpr int versionLongOpt = 1;
    constexpr int formatLongOpt = 2;
    constexpr int filterLongOpt = 3;
    constexpr int outputTimesLongOpt = 4;

    //+ not documented
    struct ::option longOptions[] = {{"help", no_argument, nullptr, 'h'},
                                     {"verifier", required_argument, nullptr, 'V'},
                                     {"input-explanations", required_argument, nullptr, 'E'},
                                     {"output-explanations", required_argument, nullptr, 'e'},
                                     {"output-stats", required_argument, nullptr, 's'},
                                     {"output-times", required_argument, &selectedLongOpt, outputTimesLongOpt},
                                     {"verbose", no_argument, nullptr, 'v'},
                                     {"quiet", no_argument, nullptr, 'q'},
                                     // {"version", no_argument, &selectedLongOpt, versionLongOpt},
                                     {"reverse-var", no_argument, nullptr, 'R'},
                                     {"format", required_argument, &selectedLongOpt, formatLongOpt},
                                     {"shuffle-samples", no_argument, nullptr, 'r'},
                                     {"max-samples", required_argument, nullptr, 'n'},
                                     {"filter-samples", required_argument, &selectedLongOpt, filterLongOpt},
                                     {0, 0, 0, 0}};

    std::string optString = ":hV:E:e:s:vqRSIrn:";

    while (true) {
        int optIndex = 0;
        int c = getopt_long(argc, argv, optString.c_str(), longOptions, &optIndex);
        if (c == -1) { break; }

        switch (c) {
            case 0: {
                std::string_view optargStr{optarg};
                switch (selectedLongOpt) {
                    case outputTimesLongOpt:
                        config.setTimesFileName(optarg);
                        break;
                    case formatLongOpt:
                        if (optargStr == "smtlib2") {
                            config.printIntervalExplanationsInSmtLib2Format();
                        } else if (optargStr == "intervals") {
                            config.printIntervalExplanationsInIntervalFormat();
                        } else {
                            assert(optargStr == "bounds");
                            config.printingIntervalExplanationsInBoundFormat();
                        }
                        break;
                    case filterLongOpt:
                        std::optional<bool> optCorrectnessFilter{};
                        if (optargStr.starts_with("in")) {
                            optargStr.remove_prefix(2);
                            optCorrectnessFilter = false;
                        }
                        if (optargStr == "correct") {
                            if (not optCorrectnessFilter.has_value()) { optCorrectnessFilter = true; }
                        } else {
                            assert(not optCorrectnessFilter.has_value());
                        }
                        if (optCorrectnessFilter.has_value()) {
                            if (*optCorrectnessFilter) {
                                config.filterCorrectSamples();
                            } else {
                                config.filterIncorrectSamples();
                            }
                            break;
                        }

                        if (optargStr.starts_with("class")) {
                            optargStr.remove_prefix(5);
                            spexplain::Network::Classification::Label l;
                            l = std::stoi(std::string{optargStr});
                            config.filterSamplesOfExpectedClass(l);
                            break;
                        }

                        assert(false);
                }
                break;
            }
            case 'h':
                printUsage(argv);
                return 0;
            case 'V':
                config.setVerifierName(optarg);
                break;
            case 'E':
                if (not explanationsFnPtr) {
                    std::cerr << "Option '-" << char(c) << "' is invalid in this context\n";
                    printUsage(argv, std::cerr);
                    return 1;
                }
                *explanationsFnPtr = optarg;
                break;
            case 'e':
                config.setExplanationsFileName(optarg);
                break;
            case 's':
                config.setStatsFileName(optarg);
                break;
            case 'v':
                config.beVerbose();
                break;
            case 'q':
                config.beQuiet();
                break;
            case 'R':
                config.reverseVarOrdering();
                break;
            case 'S':
                config.printIntervalExplanationsInSmtLib2Format();
                break;
            case 'I':
                config.printIntervalExplanationsInIntervalFormat();
                break;
            case 'r':
                config.shuffleSamples();
                break;
            case 'n': {
                auto const n = std::stoull(optarg);
                config.setMaxSamples(n);
                break;
            }
            case ':':
                std::cerr << "Option: '-" << char(optopt) << "' requires an argument\n\n";
                printUsage(argv, std::cerr);
                return 1;
            default:
                assert(c == '?');
                std::cerr << "Unrecognized option: '-" << char(optopt) << "'\n\n";
                printUsage(argv, std::cerr);
                return 1;
        }
    }

    return std::nullopt;
}

int mainExplain(int argc, char * argv[], int i, int nArgs) {
    assert(nArgs >= 1);

    constexpr int minArgs = 3;
    if (nArgs < minArgs) {
        std::cerr << "Expected at least " << minArgs << " arguments for explain, got: " << nArgs << '\n';
        printUsage(argv, std::cerr);
        return 1;
    }

    std::string_view const nnModelFn = argv[++i];
    auto networkPtr = spexplain::Network::fromNNetFile(nnModelFn);
    assert(networkPtr);

    std::string_view const datasetFn = argv[++i];

    std::string_view const strategiesSpec = argv[++i];

    std::string explanationsFn;

    spexplain::Framework::Config config;

    if (auto optRet = getOpts(argc, argv, config, &explanationsFn)) { return *optRet; }

    std::istringstream strategiesSpecIss{std::string{strategiesSpec}};
    spexplain::Framework framework{config, std::move(networkPtr), strategiesSpecIss};

    auto dataset = spexplain::Network::Dataset{datasetFn};
    std::size_t const size = dataset.size();

    spexplain::Explanations explanations =
        explanationsFn.empty() ? framework.explain(dataset) : framework.expand(explanationsFn, dataset);
    assert(explanations.size() == size);

    return 0;
}

int mainDumpPsi(int argc, char * argv[], int i, [[maybe_unused]] int nArgs) {
    assert(nArgs >= 1);

    std::string_view const nnModelFn = argv[++i];
    auto networkPtr = spexplain::Network::fromNNetFile(nnModelFn);
    assert(networkPtr);

    spexplain::Framework::Config config;

    if (auto optRet = getOpts(argc, argv, config)) { return *optRet; }

    //++ should not be necessary
    std::istringstream strategiesSpecIss{spexplain::Framework::Expand::NopStrategy::name()};
    spexplain::Framework framework{config, std::move(networkPtr), strategiesSpecIss};

    framework.dumpDomainsAsSmtLib2Query();
    framework.dumpClassificationsAsSmtLib2Queries();

    return 0;
}
} // namespace

int main(int argc, char * argv[]) try {
    constexpr int minArgs = 2;

    int nArgs = argc - 1;
    assert(nArgs >= 0);
    if (nArgs == 0) {
        printUsage(argv);
        return 0;
    }

    if (nArgs < minArgs) {
        std::cerr << "Expected at least " << minArgs << " arguments, got: " << nArgs << '\n';
        printUsage(argv, std::cerr);
        return 1;
    }

    int i = 0;
    std::string_view const maybeAction = argv[++i];
    --nArgs;

    if (maybeAction == "explain") { return mainExplain(argc, argv, i, nArgs); }
    if (maybeAction == "dump-psi") { return mainDumpPsi(argc, argv, i, nArgs); }

    // Assume the default action
    --i;
    ++nArgs;
    return mainExplain(argc, argv, i, nArgs);

} catch (std::system_error const & e) {
    std::cerr << "Terminated with a system error:\n" << e.what() << '\n' << std::endl;
    printUsage(argv, std::cerr);
    return e.code().value();
} catch (std::logic_error const & e) {
    std::cerr << "Terminated with a logic error:\n" << e.what() << '\n' << std::endl;
    printUsage(argv, std::cerr);
    return 1;
} catch (std::exception const & e) {
    std::cerr << "Terminated with an exception:\n" << e.what() << '\n' << std::endl;
    printUsage(argv, std::cerr);
    return 2;
}
