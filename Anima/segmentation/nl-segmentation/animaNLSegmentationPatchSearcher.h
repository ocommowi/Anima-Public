#pragma once

#include <animaNonLocalPatchBaseSearcher.h>

namespace anima
{

/**
 * Non local means patch searcher: computes actual patch matches weights and conformity.
 * Supports only one comparison image that is the same as the input
 */
template <class ImageType, class DataImageType, class SegmentationImageType>
class NLSegmentationPatchSearcher : public anima::NonLocalPatchBaseSearcher <ImageType, SegmentationImageType>
{
public:
    typedef typename DataImageType::Pointer DataImagePointer;
    typedef anima::NonLocalPatchBaseSearcher <ImageType> Superclass;
    typedef typename Superclass::ImageRegionType ImageRegionType;
    typedef typename Superclass::IndexType IndexType;

    NLSegmentationPatchSearcher();
    virtual ~NLSegmentationPatchSearcher() {}

    void SetThreshold(double arg) {m_Threshold = arg;}

    void SetMeanImage(DataImageType *arg) {m_MeanImage = arg;}
    void SetVarImage(DataImageType *arg) {m_VarImage = arg;}

    void AddComparisonMeanImage(DataImageType *arg);
    void ClearComparisonMeanImages() {m_ComparisonMeanImages.clear();}

    void AddComparisonVarImage(DataImageType *arg);
    void ClearComparisonVarImages() {m_ComparisonVarImages.clear();}

protected:
    virtual double ComputeWeightValue(unsigned int index, ImageRegionType &refPatch, ImageRegionType &movingPatch);
    virtual bool TestPatchConformity(unsigned int index, const IndexType &refIndex, const IndexType &movingIndex);

private:
    DataImagePointer m_MeanImage;
    DataImagePointer m_VarImage;

    std::vector <DataImagePointer> m_ComparisonMeanImages;
    std::vector <DataImagePointer> m_ComparisonVarImages;

    double m_Threshold;
};

} // end namespace anima

#include "animaNLSegmentationPatchSearcher.hxx"
