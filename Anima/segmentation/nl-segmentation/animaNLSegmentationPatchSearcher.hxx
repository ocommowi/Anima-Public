#pragma once
#include "animaNLSegmentationPatchSearcher.h"

namespace anima
{

template <class ImageType, class DataImageType, class SegmentationImageType>
NLSegmentationPatchSearcher <ImageType, DataImageType, SegmentationImageType>
::NLSegmentationPatchSearcher()
{
    m_Threshold = 0.95;
}

template <class ImageType, class DataImageType, class SegmentationImageType>
void
NLSegmentationPatchSearcher <ImageType, DataImageType, SegmentationImageType>
::AddComparisonMeanImage(DataImageType *arg)
{
    m_ComparisonMeanImages.push_back(arg);
}

template <class ImageType, class DataImageType, class SegmentationImageType>
void
NLSegmentationPatchSearcher <ImageType, DataImageType, SegmentationImageType>
::AddComparisonVarImage(DataImageType *arg)
{
    m_ComparisonVarImages.push_back(arg);
}

template <class ImageType, class DataImageType, class SegmentationImageType>
bool
NLSegmentationPatchSearcher <ImageType, DataImageType, SegmentationImageType>
::TestPatchConformity(unsigned int index, const IndexType &refIndex, const IndexType &movingIndex)
{
    double refMeanValue = m_MeanImage->GetPixel(refIndex);
    double floMeanValue = m_ComparisonMeanImages[index]->GetPixel(movingIndex);

    double refVarValue = m_VarImage->GetPixel(refIndex);
    double floVarValue = m_ComparisonVarImages[index]->GetPixel(movingIndex);

    if (((refMeanValue == 0) && (floMeanValue == 0)) ||
        ((refVarValue <= 0) && (floVarValue <= 0)))
        return true;

    double ratio = 2.0 * refMeanValue * floMeanValue / (refMeanValue * refMeanValue + floMeanValue * floMeanValue);
    ratio *= 2.0 * std::sqrt(refVarValue * floVarValue) / (refVarValue + floVarValue);

    // Should we compute the weight value of this patch ?
    if (ratio > m_Threshold)
        return true;

    return false;
}

template <class ImageType, class DataImageType, class SegmentationImageType>
double
NLSegmentationPatchSearcher <ImageType, DataImageType, SegmentationImageType>
::ComputeWeightValue(unsigned int index, ImageRegionType &refPatch, ImageRegionType &movingPatch)
{
    typedef itk::ImageRegionConstIteratorWithIndex< ImageType > InIteratorType;

    InIteratorType tmpIt (this->GetInputImage(), refPatch);
    InIteratorType tmpMovingIt (this->GetComparisonImage(index), movingPatch);

    double tmpDiffValue;

    double weightValue = 0.0;
    unsigned int numVoxels = 0;

    double minAbsoluteDiffValue = -1;
    while (!tmpIt.IsAtEnd())
    {
        tmpDiffValue = (double)tmpIt.Get() - (double)tmpMovingIt.Get();
        weightValue += tmpDiffValue * tmpDiffValue;

        if ((std::abs(tmpDiffValue) < minAbsoluteDiffValue) || (minAbsoluteDiffValue < 0))
            minAbsoluteDiffValue = std::abs(tmpDiffValue);

        ++numVoxels;
        ++tmpIt;
        ++tmpMovingIt;
    }

    // Add an epsilon for computation reason
    minAbsoluteDiffValue += 1.0e-8;

    weightValue = std::exp(- weightValue / (minAbsoluteDiffValue * minAbsoluteDiffValue * numVoxels));
    return weightValue;
}

} // end namespace anima
