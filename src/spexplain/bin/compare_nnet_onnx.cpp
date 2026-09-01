#include "spexplain/network/Dataset.h"
#include "spexplain/network/Network.h"
#include "spexplain/network/Network2.h"
#include "tsolvers/lasolver/LARefs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void printUsage(char const *program)
{
    std::cout << "Usage:\n"
              << "  " << program << " <nnet_path> <onnx_path> <dataset_csv> [tol] [limit]\n\n"
              << "Example:\n"
              << "  " << program
              << " ./data/models/heart_attack/heart_attack-50.nnet"
              << " ./data/models/heart_attack/heart_attack-50.onnx"
              << " ./data/datasets/heart_attack/heart_attack_full.csv 5e-5 0\n";
}

void printValues(char const *label, std::vector<spexplain::Float> const &values)
{
    std::cout << label << " [";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        std::cout << std::setprecision(10) << values[i];
        if (i + 1 < values.size())
            std::cout << ", ";
    }
    std::cout << "]\n";
}


std::size_t tensorSize(spexplain::NetworkLayer::Shape const & shape)
{
    if (shape.empty())
        return 0;
    std::size_t size = 1;
    for (std::size_t dim : shape)
        size *= dim;
    return size;
}

double maxAbsDiff(std::vector<spexplain::Float> const & a, std::vector<spexplain::Float> const & b)
{
    if (a.size() != b.size())
        return std::numeric_limits<double>::infinity();

    double maxDiff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        maxDiff = std::max(maxDiff, std::abs(static_cast<double>(a[i] - b[i])));
    return maxDiff;
}

bool valuesSame(std::vector<spexplain::Float> const & a,
                std::vector<spexplain::Float> const & b,
                double tol)
{
    if (a.size() != b.size())
        return false;
    return maxAbsDiff(a, b) <= tol;
}

std::vector<spexplain::Float> applySigmoid(std::vector<spexplain::Float> const & values)
{
    std::vector<spexplain::Float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        out[i] = spexplain::Float{1} / (spexplain::Float{1} + std::exp(-values[i]));
    return out;
}

std::size_t classFromValues(std::vector<spexplain::Float> const & values, bool binaryFromSigmoid)
{
    if (values.empty())
        throw std::logic_error("Cannot classify empty values");

    if (values.size() == 1)
    {
        if (binaryFromSigmoid)
            return values.front() >= spexplain::Float{0.5} ? 1u : 0u;
        return values.front() < spexplain::Float{0} ? 0u : 1u;
    }

    auto const it = std::max_element(values.begin(), values.end());
    return static_cast<std::size_t>(std::distance(values.begin(), it));
}

bool isActivationType(std::string const & type)
{
    return type == "relu" || type == "sigmoid";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    std::string_view const nnetPath = argv[1];
    std::string_view const onnxPath = argv[2];
    std::string_view const datasetPath = argv[3];
    double const tol = (argc >= 5) ? std::stod(argv[4]) : 5e-5;
    std::size_t const limit = (argc >= 6) ? static_cast<std::size_t>(std::stoull(argv[5])) : 0;

    try
    {
        auto nnet = spexplain::Network::fromNNetFile(nnetPath);
        auto onnx = spexplain::Network2::fromONNXFile(onnxPath);
        if (!nnet || !onnx)
            throw std::runtime_error("Failed to load one of the models");

        spexplain::Network::Dataset dataset(*nnet, datasetPath);

        std::size_t const nRows = (limit == 0) ? dataset.size() : std::min(limit, dataset.size());

        std::size_t const nnetInputs = nnet->nInputs();
        std::size_t const onnxInputs = tensorSize(onnx->getInputShape());
        if (onnxInputs != 0 && onnxInputs != nnetInputs)
        {
            throw std::runtime_error("Input size mismatch between nnet and onnx/network2");
        }

        bool const onnxHasFinalSigmoid =
            !onnx->getLayers().empty() && onnx->getLayers().back() && onnx->getLayers().back()->getType() == "sigmoid";
        bool const alignNnetWithSigmoid = onnxHasFinalSigmoid && nnet->nOutputs() == 1;

        std::vector<std::size_t> onnxReluActivationIndices;
        {
            std::size_t actIdx = 0;
            for (auto const & layer : onnx->getLayers())
            {
                if (!layer)
                    continue;
                if (!isActivationType(layer->getType()))
                    continue;
                if (layer->getType() == "relu")
                    onnxReluActivationIndices.push_back(actIdx);
                ++actIdx;
            }
        }

        std::size_t failures = 0;
        std::size_t classMismatchesNative = 0;
        std::size_t classMismatchesAligned = 0;
        std::size_t valueMismatchesRaw = 0;
        std::size_t valueMismatchesPreSig = 0;
        std::size_t valueMismatchesAligned = 0;
        double worstDiffRaw = -1.0;
        std::size_t worstIdxRaw = 0;
        double worstDiffPreSig = -1.0;
        std::size_t worstIdxPreSig = 0;
        double worstDiffAligned = -1.0;
        std::size_t worstIdxAligned = 0;

        std::cout << "Comparing models:\n"
                  << "  nnet: " << nnetPath << "\n"
                  << "  onnx: " << onnxPath << "\n"
                  << "  data: " << datasetPath << "\n"
                  << "  rows: " << nRows << "\n"
                  << "  tol:  " << tol << "\n"
                  << "  align nnet final sigmoid: " << (alignNnetWithSigmoid ? "yes" : "no") << "\n\n";

        for (std::size_t i = 0; i < nRows; ++i)
        {
            spexplain::Network::Sample const & sample = dataset.getSample(i);

            spexplain::Network::EvalConfig nnetEvalConf{.storeHiddenNeuronValues = true};
            auto nnetOut = nnet->evaluate(sample, nnetEvalConf);
            spexplain::Network2::Values n2Input(sample.begin(), sample.end());
            spexplain::Network2::EvalConfig onnxEvalConf{.storeHiddenNeuronValues = true};
            auto onnxOut = onnx->evaluate(n2Input, onnxEvalConf);

            std::cout << "sample " << (i + 1) << ":\n";
            std::size_t const nNnetHidden = nnetOut.hiddenNeuronInputValues.size();
            for (std::size_t l = 0; l < nNnetHidden; ++l)
            {
                bool inputSame = false;
                bool outputSame = false;
                if (l < onnxReluActivationIndices.size())
                {
                    std::size_t onnxLayerIdx = onnxReluActivationIndices[l];
                    if (onnxLayerIdx < onnxOut.hiddenNeuronInputValues.size() &&
                        onnxLayerIdx < onnxOut.hiddenNeuronOutputValues.size())
                    {
                        inputSame = valuesSame(nnetOut.hiddenNeuronInputValues[l],
                                               onnxOut.hiddenNeuronInputValues[onnxLayerIdx],
                                               tol);
                        outputSame = valuesSame(nnetOut.hiddenNeuronOutputValues[l],
                                                onnxOut.hiddenNeuronOutputValues[onnxLayerIdx],
                                                tol);
                    }
                }

                std::cout << "Layer " << (l + 1) << ":\n"
                          << "input_values: " << (inputSame ? "same" : "different") << "\n"
                          << "output_values: " << (outputSame ? "same" : "different") << "\n";
                std::cout << "Inputs: \n" ;
                printValues("nnet: ", nnetOut.hiddenNeuronInputValues[l]);
                printValues("onnx: ", onnxOut.hiddenNeuronInputValues[l]);
                std::cout << "Outputs: \n";
                printValues("nnet: ", nnetOut.hiddenNeuronOutputValues[l]);
                printValues("onnx: ", onnxOut.hiddenNeuronOutputValues[l]);

            }
            if (onnxReluActivationIndices.size() != nNnetHidden)
            {
                std::cout << "note: hidden ReLU layer count differs (nnet=" << nNnetHidden
                          << ", onnx=" << onnxReluActivationIndices.size() << ")\n";
            }

            std::vector<spexplain::Float> onnxPreSigValues = onnxOut.values;
            if (onnxHasFinalSigmoid && !onnxOut.hiddenNeuronInputValues.empty())
                onnxPreSigValues = onnxOut.hiddenNeuronInputValues.back();


            printValues("nnet values: ", nnetOut.values);
            printValues("onnx values: ", onnxOut.values);
            printValues("onnx pre-sigmoid values: ", onnxPreSigValues);

            double const diffRaw = maxAbsDiff(nnetOut.values, onnxOut.values);
            double const diffPreSig = maxAbsDiff(nnetOut.values, onnxPreSigValues);
            std::cout << "presig: " << diffPreSig << "\n";
            std::vector<spexplain::Float> nnetAlignedValues =
                alignNnetWithSigmoid ? applySigmoid(nnetOut.values) : nnetOut.values;
            double const diffAligned = maxAbsDiff(nnetAlignedValues, onnxOut.values);
            bool const classMismatch = (nnetOut.classification.label != onnxOut.classification.label);
            bool const valueMismatchRaw = (diffRaw > tol);
            bool const valueMismatchPreSig = (diffPreSig > tol);
            bool const valueMismatchAligned = (diffAligned > tol);

            std::size_t const nnetAlignedLabel = classFromValues(nnetAlignedValues, alignNnetWithSigmoid);
            std::size_t const onnxAlignedLabel = classFromValues(onnxOut.values, alignNnetWithSigmoid);
            bool const classMismatchAligned = (nnetAlignedLabel != onnxAlignedLabel);

            if (valueMismatchRaw)
                ++valueMismatchesRaw;
            if (valueMismatchPreSig)
                ++valueMismatchesPreSig;
            if (valueMismatchAligned)
                ++valueMismatchesAligned;
            if (classMismatch)
                ++classMismatchesNative;
            if (classMismatchAligned)
                ++classMismatchesAligned;

            if (diffRaw > worstDiffRaw)
            {
                worstDiffRaw = diffRaw;
                worstIdxRaw = i;
            }

            if (diffPreSig > worstDiffPreSig)
            {
                worstDiffPreSig = diffPreSig;
                worstIdxPreSig = i;
            }

            if (diffAligned > worstDiffAligned)
            {
                worstDiffAligned = diffAligned;
                worstIdxAligned = i;
            }

            if (classMismatchAligned || valueMismatchAligned)
            {
                ++failures;

                std::cout << "row[" << i << "] FAIL"
                          << " diffRaw=" << std::setprecision(10) << diffRaw
                          << " diffPreSig=" << std::setprecision(10) << diffPreSig
                          << " diffAligned=" << std::setprecision(10) << diffAligned
                          << " classNative(nnet=" << nnetOut.classification.label
                          << ", onnx=" << onnxOut.classification.label << ")"
                          << " classAligned(nnet=" << nnetAlignedLabel
                          << ", onnx=" << onnxAlignedLabel << ")\n";
            }
        }

        std::cout << "\n--- Summary ---\n"
                  << "rows checked:        " << nRows << "\n"
                  << "total failures:      " << failures << "\n"
                  << "value mismatches raw:    " << valueMismatchesRaw << "\n"
                  << "value mismatches preSig: " << valueMismatchesPreSig << "\n"
                  << "value mismatches aligned:" << valueMismatchesAligned << "\n"
                  << "class mismatches native: " << classMismatchesNative << "\n"
                  << "class mismatches aligned:" << classMismatchesAligned << "\n"
                  << "worst raw max-abs diff:      " << std::setprecision(10) << worstDiffRaw
                  << " (row " << worstIdxRaw << ")\n"
                  << "worst preSig max-abs diff:   " << std::setprecision(10) << worstDiffPreSig
                  << " (row " << worstIdxPreSig << ")\n"
                  << "worst aligned max-abs diff:  " << std::setprecision(10) << worstDiffAligned
                  << " (row " << worstIdxAligned << ")\n";

        if (failures == 0)
        {
            std::cout << "PASS: nnet and onnx/network2 match within tolerance and labels (aligned comparison).\n";
            return 0;
        }

        std::cout << "FAIL: differences detected between nnet and onnx/network2.\n";
        return 5;
    }
    catch (std::exception const &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}




