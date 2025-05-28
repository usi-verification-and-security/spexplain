#include "Dataset.h"

#include <filesystem>
#include <fstream>
#include <numeric>
#include <ostream>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>

#ifndef NDEBUG
#include <cmath>
#endif

namespace xspace {
Network::Dataset::Dataset(std::string_view fileName) {
    std::ifstream file{std::string{fileName}};
    if (not file.good()) { throw std::ifstream::failure{"Could not open dataset file "s + std::string{fileName}}; }

    // Read the first line to skip the header
    std::string header;
    getline(file, header);

    std::string line;
    Sample::Idx idx{};
    while (std::getline(file, line)) {
        std::istringstream ss{line};
        std::string field;
        Sample sample;
        while (std::getline(ss, field, ',')) {
            sample.push_back(std::stof(field));
        }
        assert(not sample.empty());
        Float expectedClassFloat = sample.back();
        assert(expectedClassFloat == std::floor(expectedClassFloat));
        sample.pop_back();
        samples.push_back(std::move(sample));

        Classification::Label label = expectedClassFloat;
        expectedClassifications.push_back({.label = label});
#ifndef NDEBUG
        classificationLabels.insert(label);
#endif

        SampleIndices & sampleIndicesOfClass = getSampleIndicesOfClass(label);
        sampleIndicesOfClass.push_back(idx);

        ++idx;
    }

    assert(not samples.empty());
    assert(size() == samples.size());
    assert(size() == expectedClassifications.size());

    assert(classificationSize() >= 2);
    assert(classificationSize() == classificationLabels.size());
    assert(classificationSize() == sampleIndicesOfClasses.size());
}

std::size_t Network::Dataset::classificationSize() const {
    return sampleIndicesOfClasses.size();
}

Network::Dataset::SampleIndices Network::Dataset::getSampleIndices() const {
    SampleIndices indices(size());
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

auto & Network::Dataset::getSampleIndicesOfClassTp(auto & vec, Classification::Label label) {
    if constexpr (not std::is_const_v<std::remove_reference_t<decltype(vec)>>) {
        std::size_t const requiredSize = label + 1;
        if (vec.size() < requiredSize) { vec.resize(requiredSize); }
    }

    return vec[label];
}

Network::Dataset::SampleIndices const & Network::Dataset::getSampleIndicesOfClass(Classification::Label label) const {
    return getSampleIndicesOfClassTp(sampleIndicesOfClasses, label);
}

Network::Dataset::SampleIndices & Network::Dataset::getSampleIndicesOfClass(Classification::Label label) {
    return getSampleIndicesOfClassTp(sampleIndicesOfClasses, label);
}

Network::Dataset::SampleIndices const &
Network::Dataset::getSampleIndicesOfExpectedClass(Classification::Label label) const {
    return getSampleIndicesOfClass(label);
}

void Network::Dataset::setComputedOutputs(Outputs outs) {
    assert(outs.size() == size());
    computedOutputs = std::move(outs);

#ifndef NDEBUG
    for (auto & output : computedOutputs) {
        assert(classificationLabels.contains(output.classification.label));
        assert(not output.values.empty());
        if (output.values.size() == 1) {
            assert(classificationSize() == 2);
        } else {
            assert(output.values.size() == classificationSize());
        }
    }
#endif

    setCorrectAndIncorrectSamples();
}

bool Network::Dataset::isCorrect(Sample::Idx idx) {
    auto const expectedLabel = getExpectedClassification(idx).label;
    auto const computedLabel = getComputedOutput(idx).classification.label;

    return expectedLabel == computedLabel;
}

void Network::Dataset::setCorrectAndIncorrectSamples() {
    std::size_t const classificationSize_ = classificationSize();
    for (std::size_t label = 0; label < classificationSize_; ++label) {
        getCorrectSampleIndicesOfClass(label);
        getIncorrectSampleIndicesOfClass(label);
    }

    std::size_t const size_ = size();
    for (Sample::Idx idx = 0; idx < size_; ++idx) {
        bool const correct = isCorrect(idx);

        auto & targetSampleIndices = correct ? correctSampleIndices : incorrectSampleIndices;
        targetSampleIndices.push_back(idx);

        auto const label = getExpectedClassification(idx).label;
        auto & targetSampleIndicesOfClass =
            correct ? getCorrectSampleIndicesOfClass(label) : getIncorrectSampleIndicesOfClass(label);
        targetSampleIndicesOfClass.push_back(idx);
    }

    assert(size() == correctSampleIndices.size() + incorrectSampleIndices.size());

    assert(classificationSize_ == correctSampleIndicesOfClasses.size());
    assert(classificationSize_ == incorrectSampleIndicesOfClasses.size());
}

Network::Dataset::SampleIndices const & Network::Dataset::getCorrectSampleIndices() const {
    return correctSampleIndices;
}

Network::Dataset::SampleIndices const & Network::Dataset::getIncorrectSampleIndices() const {
    return incorrectSampleIndices;
}

Network::Dataset::SampleIndices const &
Network::Dataset::getCorrectSampleIndicesOfClass(Classification::Label label) const {
    return getSampleIndicesOfClassTp(correctSampleIndicesOfClasses, label);
}

Network::Dataset::SampleIndices const &
Network::Dataset::getIncorrectSampleIndicesOfClass(Classification::Label label) const {
    return getSampleIndicesOfClassTp(incorrectSampleIndicesOfClasses, label);
}

Network::Dataset::SampleIndices & Network::Dataset::getCorrectSampleIndicesOfClass(Classification::Label label) {
    return getSampleIndicesOfClassTp(correctSampleIndicesOfClasses, label);
}

Network::Dataset::SampleIndices & Network::Dataset::getIncorrectSampleIndicesOfClass(Classification::Label label) {
    return getSampleIndicesOfClassTp(incorrectSampleIndicesOfClasses, label);
}

Network::Dataset::SampleIndices const &
Network::Dataset::getCorrectSampleIndicesOfExpectedClass(Classification::Label label) const {
    return getCorrectSampleIndicesOfClass(label);
}

Network::Dataset::SampleIndices const &
Network::Dataset::getIncorrectSampleIndicesOfExpectedClass(Classification::Label label) const {
    return getIncorrectSampleIndicesOfClass(label);
}
} // namespace xspace
