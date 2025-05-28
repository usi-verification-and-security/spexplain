#ifndef XSPACE_DATASET_H
#define XSPACE_DATASET_H

#include "Network.h"

#include <xspace/common/Var.h>

#include <cassert>
#include <iosfwd>
#include <string_view>

#ifndef NDEBUG
#include <unordered_set>
#endif

namespace xspace {
class Network::Dataset {
public:
    using Samples = std::vector<Sample>;
    using SampleIndices = std::vector<Sample::Idx>;
    using Outputs = std::vector<Output>;

    Dataset(std::string_view fileName);

    std::size_t size() const { return getSamples().size(); }

    std::size_t classificationSize() const;

    Samples const & getSamples() const { return samples; }
    Sample const & getSample(Sample::Idx idx) const {
        assert(idx < size());
        return samples[idx];
    }

    SampleIndices getSampleIndices() const;

    Classifications const & getExpectedClassifications() const { return expectedClassifications; }
    Classification const & getExpectedClassification(Sample::Idx idx) const {
        assert(idx < size());
        return expectedClassifications[idx];
    }

    SampleIndices const & getSampleIndicesOfExpectedClass(Classification::Label) const;

    void setComputedOutputs(Outputs);

    Outputs const & getComputedOutputs() const { return computedOutputs; }
    Output const & getComputedOutput(Sample::Idx idx) const {
        assert(idx < size());
        return computedOutputs[idx];
    }

    bool isCorrect(Sample::Idx);

    //+ can also be a span
    SampleIndices const & getCorrectSampleIndices() const;
    SampleIndices const & getIncorrectSampleIndices() const;

    SampleIndices const & getCorrectSampleIndicesOfExpectedClass(Classification::Label) const;
    SampleIndices const & getIncorrectSampleIndicesOfExpectedClass(Classification::Label) const;

protected:
    void setCorrectAndIncorrectSamples();

    // The original order of the samples should remain unchanged
    Samples samples{};

    Classifications expectedClassifications{};

    Outputs computedOutputs{};

private:
    static auto & getSampleIndicesOfClassTp(auto &, Classification::Label);

    SampleIndices const & getSampleIndicesOfClass(Classification::Label) const;
    SampleIndices & getSampleIndicesOfClass(Classification::Label);

    SampleIndices const & getCorrectSampleIndicesOfClass(Classification::Label) const;
    SampleIndices const & getIncorrectSampleIndicesOfClass(Classification::Label) const;
    SampleIndices & getCorrectSampleIndicesOfClass(Classification::Label);
    SampleIndices & getIncorrectSampleIndicesOfClass(Classification::Label);

    SampleIndices correctSampleIndices{};
    SampleIndices incorrectSampleIndices{};

    std::vector<SampleIndices> sampleIndicesOfClasses{};

    std::vector<SampleIndices> correctSampleIndicesOfClasses{};
    std::vector<SampleIndices> incorrectSampleIndicesOfClasses{};

#ifndef NDEBUG
    std::unordered_set<Classification::Label> classificationLabels{};
#endif
};
} // namespace xspace

#endif // XSPACE_DATASET_H
