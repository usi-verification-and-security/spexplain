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
void printUsageStrategyRow(std::ostream & os, std::string_view name, std::vector<std::string_view> params = {},
                           std::vector<std::string_view> defaultParams = {}) {
    os << std::setw(12) << name;
    if (not params.empty()) {
        os << ": " << params.front();
        for (auto param : params | std::views::drop(1)) {
            os << ", " << param;
        }
        if (not defaultParams.empty()) {
            os << " (default: " << defaultParams.front();
            for (auto param : defaultParams | std::views::drop(1)) {
                os << ", " << param;
            }
            os << ')';
        }
    }
    os << '\n';
}

constexpr int optMaxWidth = 7;
constexpr int optArgMaxWidth = 7;

void printUsageOptRow(std::ostream & os, char opt, std::string_view arg = "", std::string_view desc = "") {
    os << "    -" << opt << ' ';
    os << std::left << std::setw(optArgMaxWidth);
    if (arg.empty()) {
        os << ' ';
    } else {
        os << arg;
        if (arg.size() >= optArgMaxWidth) { os << '\n' << std::left << std::setw(optMaxWidth + optArgMaxWidth) << ' '; }
    }
    os << desc << '\n';
}

void printUsageLongOptRow(std::ostream & os, std::string_view longOpt, std::string_view arg = "",
                          std::string_view desc = "") {
    os << "    --" << longOpt << " " << arg << '\n';
    if (desc.empty()) { return; }

    os << std::left << std::setw(optMaxWidth + optArgMaxWidth) << ' ' << desc << '\n';
}

bool stringViewToBool(std::string_view sw) {
    bool val;
    std::string str{sw};
    std::istringstream iss{str};
    iss >> std::boolalpha >> val;
    if (not iss) { throw std::invalid_argument{"Expected a boolean, got: "s + str}; }
    return val;
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
    os << "\t explain:\t <nn_model_fn> <dataset_fn> [<exp_strategies_spec>]\n";
    os << "\t dump-psi:\t <nn_model_fn>\n";

    os << "STRATEGIES SPEC: '<spec1>[; <spec2>]...'\n";
    os << "Each spec: '<name>[ <param>[, <param>]...]'\n";
    os << "Strategies and possible parameters:\n";
    //+ template by the strategy and move the params to the classes as well
    printUsageStrategyRow(os, Framework::Expand::NopStrategy::name());
    printUsageStrategyRow(os, Framework::Expand::AbductiveStrategy::name());
    //+ also include 'vars'
    printUsageStrategyRow(os, Framework::Expand::TrialAndErrorStrategy::name(), {"n <int>"}, {"n 4"});
    printUsageStrategyRow(os, UnsatCoreStrategy::name(), {"interval", "min", "vars x<i>..."});
    printUsageStrategyRow(os, InterpolationStrategy::name(),
                          {"weak", "strong", "weaker", "stronger", "bweak", "bstrong", "aweak", "astrong", "aweaker",
                           "astronger", "afactor <factor>", "vars x<i>..."},
                          {"aweak", "bstrong"});
    printUsageStrategyRow(os, Framework::Expand::SliceStrategy::name(), {"[vars] x<i>..."});
    os << "Default strategy: " << InterpolationStrategy::name() << '\n';

    os << "VERIFIERS: opensmt";
#ifdef MARABOU
    os << " marabou";
#endif
    os << '\n';

    os << "OPTIONS:\n";
    printUsageLongOptRow(os, "help");
    printUsageOptRow(os, 'h', "", "Prints this help message and exits");
    printUsageLongOptRow(os, "verifier");
    printUsageOptRow(os, 'V', "<name>", "Set the verifier");
    printUsageLongOptRow(os, "input-explanations");
    printUsageOptRow(os, 'E', "<file>", "Use explanations from the file as starting points");
    printUsageLongOptRow(os, "output-explanations");
    printUsageOptRow(os, 'e', "<file>",
                     "Output explanations into the file (default: "s + Framework::Config::defaultExplanationsFileName +
                         ")");
    printUsageLongOptRow(os, "output-stats");
    printUsageOptRow(os, 's', "<file>", "Output statistics into the file");
    printUsageLongOptRow(os, "output-times", "<file>", "Output runtime splits into the file");
    printUsageLongOptRow(os, "verbose");
    printUsageOptRow(os, 'v', "", "Run in verbose mode");
    printUsageLongOptRow(os, "quiet");
    printUsageOptRow(os, 'q', "", "Run in quiet mode");
    printUsageLongOptRow(os, "reverse-var");
    printUsageOptRow(os, 'R', "", "Reverse the order of variables");
    printUsageLongOptRow(os, "encoding-neuron-vars", "true|false",
                         "Encode the internal neurons using (or not) auxiliary variables");
    printUsageLongOptRow(os, "encoding-output-vars", "true|false",
                         "Encode the output neurons using (or not) auxiliary variables");
    printUsageLongOptRow(os, "encoding-relu-lower-bounds", "true|false",
                         "Encode ReLUs using (or not) firm lower bounds");
    printUsageLongOptRow(os, "allow-neuron-vars-in-explanations", "true|false",
                         "Allow (or not) variables of internal neurons in explanations (regardless of used encoding)");
    printUsageLongOptRow(
        os, "fix-default-sample-neuron-activations", "all|none|[in]active",
        "Set default fixing of given sample-based neuron activations (default: "s +
            defaultSampleNeuronActivationsToString(Framework::Config::defaultFixingOfSampleNeuronActivations) + ")");
    printUsageLongOptRow(
        os, "prefer-default-sample-neuron-activations", "all|none|[in]active",
        "Set default preference of given sample-based neuron activations (default: "s +
            defaultSampleNeuronActivationsToString(Framework::Config::defaultPreferenceOfSampleNeuronActivations) +
            ")");
    printUsageLongOptRow(os, "fix-all-sample-neuron-activations-at", "<l>,[-]<n>",
                         "Fix (or not with '-') sample neuron activation for given layer and neuron but all samples");
    printUsageLongOptRow(
        os, "prefer-all-sample-neuron-activations-at", "<l>,[-]<n>",
        "Prefer (or not with '-') sample neuron activation for given layer and neuron but all samples");
    printUsageLongOptRow(os, "fix-sample-neuron-activation-at", "<idx>,<l>,[-]<n>",
                         "Fix (or not with '-') sample neuron activation for given sample, layer and neuron");
    printUsageLongOptRow(os, "prefer-sample-neuron-activation-at", "<idx>,<l>,[-]<n>",
                         "Prefer (or not with '-') sample neuron activation for given sample, layer and neuron");
    printUsageLongOptRow(os, "input-fix-sample-neuron-activations", "<file>",
                         "Fix sample neuron activations as specified in the file");
    printUsageLongOptRow(os, "input-prefer-sample-neuron-activations", "<file>",
                         "Prefer sample neuron activations as specified in the file");
    printUsageOptRow(os, 'S', "", "Print the resulting explanations in the SMT-LIB2 format");
    printUsageOptRow(os, 'I', "", "Print the resulting explanations in the form of intervals");
    printUsageLongOptRow(os, "format", "smtlib2|intervals|bounds", "Use one of the output explanation formats");
    printUsageLongOptRow(os, "shuffle-samples");
    printUsageOptRow(os, 'r', "", "Shuffle (randomize) samples");
    printUsageLongOptRow(os, "max-samples");
    printUsageOptRow(os, 'n', "<int>", "Maximum no. samples to be processed");
    printUsageLongOptRow(os, "samples");
    //+ change s.t. '<idx>' is just one sample and '<idx>,' starts from the sample
    printUsageOptRow(os, 'i', "<idx>[,<idx2>]", "Only process samples starting from <idx> [and ending at <idx2>]");
    printUsageLongOptRow(os, "filter-samples", "[in]correct|class<c>",
                         "Only process sample points that match the given filter");
    printUsageLongOptRow(os, "time-limit-per");
    printUsageOptRow(os, 't', "<ms>", "Time limit per explanation in miliseconds");

    os << "\nEXAMPLES:\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv\n";
    os << cmd << " explain data/models/toy.nnet data/datasets/toy.csv abductive -e data/explanations/toy.phi.txt\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'ucore interval, min' -RS -e toy.phi.txt\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'itp aweaker, bstrong; ucore'\n";
    os << cmd << " data/models/toy.nnet data/datasets/toy.csv 'trial n 3' -n2 -s stats.txt\n";
    os << cmd << " dump-psi data/models/toy.nnet\n";

    os.flush();
}

std::optional<int> getOpts(int argc, char * argv[], spexplain::Framework::Config & config,
                           spexplain::Network const & network, std::string * explanationsFnPtr = nullptr) {

    int selectedLongOpt = 0;
    // constexpr int versionLongOpt = 1;
    constexpr int formatLongOpt = 2;
    constexpr int filterLongOpt = 3;
    constexpr int outputTimesLongOpt = 4;
    constexpr int fixDefaultSampleNeuronActivationsLongOpt = 5;
    constexpr int preferDefaultSampleNeuronActivationsLongOpt = 6;
    constexpr int fixAllSampleNeuronActivationsLongOpt = 7;
    constexpr int preferAllSampleNeuronActivationsLongOpt = 8;
    constexpr int fixSampleNeuronActivationLongOpt = 9;
    constexpr int preferSampleNeuronActivationLongOpt = 10;
    constexpr int inputFixSampleNeuronActivationsLongOpt = 11;
    constexpr int inputPreferSampleNeuronActivationsLongOpt = 12;
    constexpr int encodingNeuronVarsLongOpt = 13;
    constexpr int encodingOutputVarsLongOpt = 14;
    constexpr int encodingReluLowerBoundsLongOpt = 15;
    constexpr int allowNeuronVarsInExplanationsLongOpt = 16;

    struct ::option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"verifier", required_argument, nullptr, 'V'},
        {"input-explanations", required_argument, nullptr, 'E'},
        {"output-explanations", required_argument, nullptr, 'e'},
        {"output-stats", required_argument, nullptr, 's'},
        {"output-times", required_argument, &selectedLongOpt, outputTimesLongOpt},
        {"verbose", no_argument, nullptr, 'v'},
        {"quiet", no_argument, nullptr, 'q'},
        // {"version", no_argument, &selectedLongOpt, versionLongOpt},
        {"reverse-var", no_argument, nullptr, 'R'},
        {"encoding-neuron-vars", required_argument, &selectedLongOpt, encodingNeuronVarsLongOpt},
        {"encoding-output-vars", required_argument, &selectedLongOpt, encodingOutputVarsLongOpt},
        {"encoding-relu-lower-bounds", required_argument, &selectedLongOpt, encodingReluLowerBoundsLongOpt},
        {"allow-neuron-vars-in-explanations", required_argument, &selectedLongOpt,
         allowNeuronVarsInExplanationsLongOpt},
        {"fix-default-sample-neuron-activations", required_argument, &selectedLongOpt,
         fixDefaultSampleNeuronActivationsLongOpt},
        {"prefer-default-sample-neuron-activations", required_argument, &selectedLongOpt,
         preferDefaultSampleNeuronActivationsLongOpt},
        {"fix-all-sample-neuron-activations-at", required_argument, &selectedLongOpt,
         fixAllSampleNeuronActivationsLongOpt},
        {"prefer-all-sample-neuron-activations-at", required_argument, &selectedLongOpt,
         preferAllSampleNeuronActivationsLongOpt},
        {"fix-sample-neuron-activation-at", required_argument, &selectedLongOpt, fixSampleNeuronActivationLongOpt},
        {"prefer-sample-neuron-activation-at", required_argument, &selectedLongOpt,
         preferSampleNeuronActivationLongOpt},
        {"input-fix-sample-neuron-activations", required_argument, &selectedLongOpt,
         inputFixSampleNeuronActivationsLongOpt},
        {"input-prefer-sample-neuron-activations", required_argument, &selectedLongOpt,
         inputPreferSampleNeuronActivationsLongOpt},
        {"format", required_argument, &selectedLongOpt, formatLongOpt},
        {"shuffle-samples", no_argument, nullptr, 'r'},
        {"max-samples", required_argument, nullptr, 'n'},
        {"samples", required_argument, nullptr, 'i'},
        {"filter-samples", required_argument, &selectedLongOpt, filterLongOpt},
        {"time-limit-per", required_argument, nullptr, 't'},
        {0, 0, 0, 0}};

    std::string optString = ":hV:E:e:s:vqRSIrn:i:t:";

    config.init(network);

    while (true) {
        int optIndex = 0;
        int c = getopt_long(argc, argv, optString.c_str(), longOptions, &optIndex);
        if (c == -1) { break; }

        switch (c) {
            case 0: {
                bool const hasArgument = optarg;
                std::string_view optargStr;
                if (hasArgument) { optargStr = optarg; }
                switch (selectedLongOpt) {
                    case outputTimesLongOpt:
                        config.setTimesFileName(optargStr);
                        break;
                    case formatLongOpt:
                        if (optargStr == "smtlib2") {
                            config.printIntervalExplanationsInSmtLib2Format();
                        } else if (optargStr == "intervals") {
                            config.printIntervalExplanationsInIntervalFormat();
                        } else {
                            assert(optargStr == "bounds");
                            config.printIntervalExplanationsInBoundFormat();
                        }
                        break;
                    case filterLongOpt: {
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
                        break;
                    }
                    case encodingNeuronVarsLongOpt:
                        assert(hasArgument);
                        config.setEncodingNeuronVars(stringViewToBool(optargStr));
                        break;
                    case encodingOutputVarsLongOpt:
                        assert(hasArgument);
                        config.setEncodingOutputVars(stringViewToBool(optargStr));
                        break;
                    case encodingReluLowerBoundsLongOpt:
                        assert(hasArgument);
                        config.setEncodingReluLowerBounds(stringViewToBool(optargStr));
                        break;
                    case allowNeuronVarsInExplanationsLongOpt:
                        assert(hasArgument);
                        config.allowNeuronVarsInExplanations(stringViewToBool(optargStr));
                        break;
                    case fixDefaultSampleNeuronActivationsLongOpt:
                        config.fixDefaultSampleNeuronActivations(
                            spexplain::makeDefaultSampleNeuronActivations(optargStr));
                        break;
                    case preferDefaultSampleNeuronActivationsLongOpt:
                        config.preferDefaultSampleNeuronActivations(
                            spexplain::makeDefaultSampleNeuronActivations(optargStr));
                        break;
                    case fixAllSampleNeuronActivationsLongOpt: {
                        auto pos = spexplain::makeHiddenNeuronPosition(optargStr);
                        config.fixAllSampleNeuronActivationsAt(pos);
                        break;
                    }
                    case preferAllSampleNeuronActivationsLongOpt: {
                        auto pos = spexplain::makeHiddenNeuronPosition(optargStr);
                        config.preferAllSampleNeuronActivationsAt(pos);
                        break;
                    }
                    case fixSampleNeuronActivationLongOpt: {
                        auto [idx, pos] = spexplain::makeHiddenNeuronPositionOfSample(optargStr);
                        config.fixSampleNeuronActivationAt(idx, pos);
                        break;
                    }
                    case preferSampleNeuronActivationLongOpt: {
                        auto [idx, pos] = spexplain::makeHiddenNeuronPositionOfSample(optargStr);
                        config.preferSampleNeuronActivationAt(idx, pos);
                        break;
                    }
                    case inputFixSampleNeuronActivationsLongOpt:
                        spexplain::parseFixSampleNeuronActivations(config, optargStr);
                        break;
                    case inputPreferSampleNeuronActivationsLongOpt:
                        spexplain::parsePreferSampleNeuronActivations(config, optargStr);
                        break;
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
            case 'i': {
                std::istringstream iss{optarg};
                std::size_t idx;
                iss >> idx;
                if (iss.fail()) {
                    std::cerr << "Option '-" << char(c) << "': parsing the first index failed\n";
                    printUsage(argv, std::cerr);
                    return 1;
                }
                config.setFirstSampleIdx(idx);
                if (not iss.eof()) {
                    char c2;
                    iss >> c2 >> idx;
                    if (iss.fail() || c2 != ',') {
                        std::cerr << "Option '-" << char(c) << "': parsing the second index failed\n";
                        printUsage(argv, std::cerr);
                        return 1;
                    }
                    if (not iss.eof()) {
                        std::cerr << "Option '-" << char(c)
                                  << "': additional arguments after parsing the second index\n";
                        printUsage(argv, std::cerr);
                        return 1;
                    }
                    config.setLastSampleIdx(idx);
                }
                break;
            }
            case 't': {
                auto const limit = std::stoull(optarg);
                config.setTimeLimitPerExplanation(limit);
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

    constexpr int minArgs = 2;
    if (nArgs < minArgs) {
        std::cerr << "Expected at least " << minArgs << " arguments for explain, got: " << nArgs << '\n';
        printUsage(argv, std::cerr);
        return 1;
    }

    std::string_view const nnModelFn = argv[++i];
    auto networkPtr = spexplain::Network::fromNNetFile(nnModelFn);
    assert(networkPtr);

    std::string_view const datasetFn = argv[++i];

    std::string_view const strategiesSpec = (i + 1 < argc) ? argv[++i] : "";

    std::string explanationsFn;

    spexplain::Framework::Config config;

    if (auto optRet = getOpts(argc, argv, config, *networkPtr, &explanationsFn)) { return *optRet; }

    std::istringstream strategiesSpecIss{std::string{strategiesSpec}};
    spexplain::Framework framework{config, std::move(networkPtr), strategiesSpecIss};

    auto dataset = spexplain::Network::Dataset{framework.getNetwork(), datasetFn};
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

    if (auto optRet = getOpts(argc, argv, config, *networkPtr)) { return *optRet; }

    spexplain::Framework framework{config};
    framework.setNetwork(std::move(networkPtr));
    framework.setVerifier();

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
