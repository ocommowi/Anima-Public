#pragma once

#include <itkImage.h>
#include <itkMacro.h>

namespace anima
{

/**
 * Abstract class for non local patch matching. May be used to search in multiple images
 * if the concrete class does support it. Does not compute weights and samples for central voxels
 * as these would always get a weight of 1. Instead, it is the developer task to implement it in his filter
 */
template <class ImageType, class SegmentationImageType = ImageType>
class NonLocalPatchBaseSearcher
{
public:
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::RegionType ImageRegionType;
    typedef typename ImageType::Pointer ImagePointer;
    typedef typename ImageType::PixelType PixelType;
    typedef typename SegmentationImageType::Pointer SegmentationImagePointer;
    typedef typename SegmentationImageType::PixelType SegmentationPixelType;

    NonLocalPatchBaseSearcher();
    virtual ~NonLocalPatchBaseSearcher() {}

    void SetPatchHalfSize(unsigned int arg) {m_PatchHalfSize = arg;}
    void SetSearchStepSize(unsigned int arg) {m_SearchStepSize = arg;}
    void SetMaxAbsDisp(unsigned int arg) {m_MaxAbsDisp = arg;}
    void SetWeightThreshold(double arg) {m_WeightThreshold = arg;}

    void SetInputImage(ImageType *arg) {m_InputImage = arg;}
    itkGetObjectMacro(InputImage, ImageType)

    void AddComparisonImage(ImageType *arg);
    ImageType *GetComparisonImage(unsigned int index);

    void AddSegmentationImage(SegmentationImageType *arg);

    itkGetConstReferenceMacro(DatabaseWeights, std::vector <double>)
    itkGetConstReferenceMacro(DatabaseSamples, std::vector <PixelType>)
    itkGetConstReferenceMacro(DatabaseSegmentationSamples, std::vector <SegmentationPixelType>)

    void UpdateAtPosition(const IndexType &dataIndex);

protected:
    virtual void ComputeInputProperties(const IndexType &refIndex, ImageRegionType &refPatch) {}
    virtual void ComputeComparisonProperties(unsigned int index, ImageRegionType &movingPatch) {}
    virtual double ComputeWeightValue(unsigned int index, ImageRegionType &refPatch, ImageRegionType &movingPatch) = 0;
    virtual bool TestPatchConformity(unsigned int index, const IndexType &refIndex, const IndexType &movingIndex) = 0;

private:
    unsigned int m_PatchHalfSize;
    unsigned int m_SearchStepSize;
    unsigned int m_MaxAbsDisp;
    double m_WeightThreshold;

    ImagePointer m_InputImage;
    std::vector <ImagePointer> m_ComparisonImages;
    std::vector <SegmentationImagePointer> m_SegmentationImages;

    std::vector <double> m_DatabaseWeights;
    std::vector <PixelType> m_DatabaseSamples;
    std::vector <SegmentationPixelType> m_DatabaseSegmentationSamples;
};

} // end namespace anima

#include "animaNonLocalPatchBaseSearcher.hxx"
