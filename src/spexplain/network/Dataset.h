#ifndef SPEXPLAIN_DATASET_H
#define SPEXPLAIN_DATASET_H

#include "Network.h"

#include <spexplain/common/Var.h>

#include <cassert>
#include <iosfwd>
#include <string_view>

#ifndef NDEBUG
#include <set>
#endif

namespace spexplain {
class Network::Dataset {
public:
    using Samples = std::vector<Sample>;
    using SampleIndices = std::vector<Sample::Idx>;
    using Outputs = std::vector<Output>;

    Dataset(Network const &, std::string_view fileName);
    Dataset(std::size_t nInputs_, std::size_t nClasses_, std::string_view fileName);

    std::size_t nInputs() const { return _nInputs; }
    std::size_t nClasses() const { return _nClasses; }

    std::size_t size() const { return getSamples().size(); }

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

    std::size_t _nInputs;
    std::size_t _nClasses;

    SampleIndices correctSampleIndices{};
    SampleIndices incorrectSampleIndices{};

    std::vector<SampleIndices> sampleIndicesOfClasses{};

    std::vector<SampleIndices> correctSampleIndicesOfClasses{};
    std::vector<SampleIndices> incorrectSampleIndicesOfClasses{};

#ifndef NDEBUG
    std::set<Classification::Label> expectedClassificationLabels{};
#endif
};
} // namespace spexplain

#endif // SPEXPLAIN_DATASET_H
