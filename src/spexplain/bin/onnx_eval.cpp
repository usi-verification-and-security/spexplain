#include "spexplain/network/Network2.h"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t tensorSize(spexplain::NetworkLayer::Shape const &shape)
{
    if (shape.empty())
        return 0;

    std::size_t size = 1;
    for (std::size_t dim : shape)
        size *= dim;

    return size;
}

std::vector<spexplain::Float> parseCsvInput(std::string const &csv)
{
    std::vector<spexplain::Float> values;
    std::stringstream ss(csv);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        if (!token.empty())
            values.push_back(std::stod(token));
    }

    return values;
}

std::vector<spexplain::Float> makeDefaultInput(std::size_t size)
{
    std::vector<spexplain::Float> input(size, 0.0);
    for (std::size_t i = 0; i < size; ++i)
        input[i] = static_cast<spexplain::Float>(i + 1) / static_cast<spexplain::Float>(10);
    return input;
}

void printShape(char const *label, spexplain::NetworkLayer::Shape const &shape)
{
    std::cout << label << " [";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i + 1 < shape.size())
            std::cout << ", ";
    }
    std::cout << "]\n";
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

void printUsage(char const *program)
{
    std::cout << "Usage:\n"
              << "  " << program << " <onnx_path> [input_csv]\n\n"
              << "Examples:\n"
              << "  " << program << " ../data/models/onnx/fc1.onnx\n"
              << "  " << program << " ../data/models/onnx/fc1.onnx 0.1,0.2,0.3,0.4\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    std::string const onnxPath = argv[1];

    try
    {
        auto network = spexplain::Network2::fromONNXFile(onnxPath);
        if (!network)
            throw std::runtime_error("Failed to create Network2 object");

        std::cout << "ONNX is parsed successfully!\n";
        std::cout << "Network2 built with " << network->getLayers().size() << " layers.\n";

        auto const &inputShape = network->getInputShape();
        auto const &outputShape = network->getOutputShape();
        std::size_t const expectedInputSize = tensorSize(inputShape);

        std::vector<spexplain::Float> input;
        if (argc >= 3)
            input = parseCsvInput(argv[2]);
        else
            input = makeDefaultInput(expectedInputSize);

        if (expectedInputSize == 0)
            throw std::runtime_error("Parsed network has empty input shape");

        if (input.size() != expectedInputSize)
        {
            std::ostringstream oss;
            oss << "Input size mismatch. Expected " << expectedInputSize << " values, got " << input.size();
            throw std::runtime_error(oss.str());
        }

        auto output = network->evaluate(input);

        printShape("Input shape:", inputShape);
        printShape("Output shape:", outputShape);
        printValues("Input:", input);
        printValues("Output:", output.values);
        std::cout << "classification (label): " << output.classification.label << "\n";
    }
    catch (std::exception const &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
