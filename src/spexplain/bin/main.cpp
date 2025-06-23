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
    printUsageStrategyRow(os, Framework::Expand::TrialAndErrorStrategy::name(), {"n <int>"});
    printUsageStrategyRow(os, UnsatCoreStrategy::name(), {"sample", "interval", "min"});
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
    printUsageOptRow(os, 'E', "<file>", "Use explanations from file as starting points");
    printUsageOptRow(os, 'v', "", "Run in verbose mode");
    printUsageOptRow(os, 'r', "", "Reverse the order of variables");
    printUsageOptRow(os, 's', "", "Print the resulting explanations in the SMT-LIB2 format");
    printUsageOptRow(os, 'i', "", "Print the resulting explanations in the form of intervals");
    printUsageOptRow(os, 'n', "<int>", "Maximum no. samples to be processed");
    printUsageOptRow(os, 'S', "", "Shuffle samples");

    os << "\nEXAMPLES:\n";
    os << cmd << " explain data/models/toy.nnet data/datasets/toy.csv abductive\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'ucore interval, min' -rvs\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'itp aweaker, bstrong; ucore'\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'trial n 2' -n1\n";
    os << cmd << " dump-psi data/models/toy.nnet\n";

    os.flush();
}

std::optional<int> getOpts(int argc, char * argv[], spexplain::Framework::Config & config, std::string & verifierName,
                           std::string & explanationsFn) {

    int selectedLongOpt = 0;
    // constexpr int versionLongOpt = 1;
    constexpr int formatLongOpt = 2;
    constexpr int filterLongOpt = 3;

    //+ not documented
    struct ::option longOptions[] = {{"help", no_argument, nullptr, 'h'},
                                     {"verifier", required_argument, nullptr, 'V'},
                                     {"input-explanations", required_argument, nullptr, 'E'},
                                     {"verbose", no_argument, nullptr, 'v'},
                                     // {"version", no_argument, &selectedLongOpt, versionLongOpt},
                                     {"reverse-var", no_argument, nullptr, 'r'},
                                     {"format", required_argument, &selectedLongOpt, formatLongOpt},
                                     {"shuffle-samples", no_argument, nullptr, 'S'},
                                     {"max-samples", required_argument, nullptr, 'n'},
                                     {"filter-samples", required_argument, &selectedLongOpt, filterLongOpt},
                                     {0, 0, 0, 0}};

    while (true) {
        int optIndex = 0;
        int c = getopt_long(argc, argv, ":hV:E:vrsiSn:", longOptions, &optIndex);
        if (c == -1) { break; }

        switch (c) {
            case 0: {
                std::string_view optargStr{optarg};
                switch (selectedLongOpt) {
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
                verifierName = optarg;
                break;
            case 'E':
                explanationsFn = optarg;
                break;
            case 'v':
                config.beVerbose();
                break;
            case 'r':
                config.reverseVarOrdering();
                break;
            case 's':
                config.printIntervalExplanationsInSmtLib2Format();
                break;
            case 'i':
                config.printIntervalExplanationsInIntervalFormat();
                break;
            case 'S':
                config.shuffleSamples();
                break;
            case 'n': {
                auto const n = std::stoull(optarg);
                config.setMaxSamples(n);
                break;
            }
            default:
                assert(c == '?');
                std::cerr << "Unrecognized option: '-" << char(optopt) << "'\n\n";
                printUsage(argv, std::cerr);
                return 1;
        }
    }

    return std::nullopt;
}

int mainExplain(int argc, char * argv[], int i) {
    std::string_view const nnModelFn = argv[++i];
    auto networkPtr = spexplain::Network::fromNNetFile(nnModelFn);
    assert(networkPtr);

    std::string_view const datasetFn = argv[++i];

    std::string_view const strategiesSpec = argv[++i];

    std::string verifierName;
    std::string explanationsFn;

    spexplain::Framework::Config config;

    if (auto optRet = getOpts(argc, argv, config, verifierName, explanationsFn)) { return *optRet; }

    auto dataset = spexplain::Network::Dataset{datasetFn};
    std::size_t const size = dataset.size();

    std::istringstream strategiesSpecIss{std::string{strategiesSpec}};
    spexplain::Framework framework{config, std::move(networkPtr), verifierName, strategiesSpecIss};

    spexplain::Explanations explanations =
        explanationsFn.empty() ? framework.explain(dataset) : framework.expand(explanationsFn, dataset);
    assert(explanations.size() == size);

    return 0;
}

int mainDumpPsi(int argc, char * argv[], int i) {
    std::string_view const nnModelFn = argv[++i];
    auto networkPtr = spexplain::Network::fromNNetFile(nnModelFn);
    assert(networkPtr);

    std::string verifierName;
    //+ does not make sense here
    std::string explanationsFn;

    spexplain::Framework::Config config;

    if (auto optRet = getOpts(argc, argv, config, verifierName, explanationsFn)) { return *optRet; }

    //++ should not be necessary
    std::istringstream strategiesSpecIss{spexplain::Framework::Expand::NopStrategy::name()};
    spexplain::Framework framework{config, std::move(networkPtr), verifierName, strategiesSpecIss};

    framework.dumpClassificationsAsSmtLib2Queries();

    return 0;
}
} // namespace

int main(int argc, char * argv[]) try {
    constexpr int minArgs = 2;

    int const nArgs = argc - 1;
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

    if (maybeAction == "explain") { return mainExplain(argc, argv, i); }
    if (maybeAction == "dump-psi") { return mainDumpPsi(argc, argv, i); }

    // Assume the default action
    --i;
    return mainExplain(argc, argv, i);

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
