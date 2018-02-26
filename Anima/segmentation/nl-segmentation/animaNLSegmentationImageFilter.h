#pragma once

#include <iostream>
#include <animaMaskedImageToImageFilter.h>
#include <itkImage.h>

#include <vector>

namespace anima
{

template <class PixelScalarType, class PixelOutputScalarType>
class NLSegmentationImageFilter :
        public anima::MaskedImageToImageFilter< itk::Image <PixelScalarType, 3> , itk::Image <PixelOutputScalarType, 3> >
{
public:
    /** Standard class typedefs. */
    typedef NLSegmentationImageFilter <PixelScalarType, PixelOutputScalarType> Self;
    typedef itk::SmartPointer <Self> Pointer;
    typedef itk::SmartPointer <const Self> ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods) */
    itkTypeMacro(NLSegmentationImageFilter, MaskedImageToImageFilter)

    /** Image typedef support */
    typedef itk::Image <PixelScalarType, 3> InputImageType;
    typedef itk::Image <PixelOutputScalarType, 3> OutputImageType;
    typedef itk::Image <double, 3> DataImageType;

    typedef typename InputImageType::PixelType VectorType;

    typedef vnl_matrix <double> CovarianceType;

    typedef typename InputImageType::Pointer InputImagePointer;
    typedef typename InputImageType::IndexType InputImageIndexType;
    typedef typename OutputImageType::Pointer OutputImagePointer;
    typedef typename DataImageType::Pointer DataImagePointer;

    /** Superclass typedefs. */
    typedef anima::MaskedImageToImageFilter< InputImageType, OutputImageType > Superclass;
    typedef typename Superclass::OutputImageRegionType OutputImageRegionType;
    typedef typename Superclass::MaskImageType MaskImageType;

    typedef struct
    {
        OutputImageRegionType movingRegion;
        unsigned int numDatabaseImage;
        InputImageIndexType centerIndex;
    } SampleIdentifier;

    void AddDatabaseInput(InputImageType *tmpIm)
    {
        m_DatabaseImages.push_back(tmpIm);
    }

    void AddDatabaseSegmentationInput(OutputImageType *tmpIm)
    {
        m_DatabaseSegmentationImages.push_back(tmpIm);
    }

    itkSetMacro(PatchHalfSize, unsigned int)
    itkSetMacro(SearchNeighborhood, unsigned int)
    itkSetMacro(SearchStepSize, unsigned int)
    itkSetMacro(WeightThreshold, double)
    itkSetMacro(Threshold, double)

protected:
    NLSegmentationImageFilter()
        : Superclass()
    {
        m_DatabaseImages.clear();
        m_DatabaseSegmentationImages.clear();

        m_WeightThreshold = 0.0;
        m_Threshold = 0.95;
        m_PatchHalfSize = 1;
        m_SearchStepSize = 2;
        m_SearchNeighborhood = 4;
    }

    virtual ~NLSegmentationImageFilter() {}

    void BeforeThreadedGenerateData() ITK_OVERRIDE;
    void ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, itk::ThreadIdType threadId) ITK_OVERRIDE;

    void CheckComputationMask() ITK_OVERRIDE;

private:
    ITK_DISALLOW_COPY_AND_ASSIGN(NLSegmentationImageFilter);

    std::vector <InputImagePointer> m_DatabaseImages;
    std::vector <OutputImagePointer> m_DatabaseSegmentationImages;

    std::vector <DataImagePointer> m_DatabaseMeanImages;
    std::vector <DataImagePointer> m_DatabaseVarImages;
    DataImagePointer m_ReferenceMeanImage;
    DataImagePointer m_ReferenceVarImage;

    double m_WeightThreshold;
    double m_Threshold;

    unsigned int m_PatchHalfSize;
    unsigned int m_SearchStepSize;
    unsigned int m_SearchNeighborhood;
};

} // end namespace anima

#include "animaNLSegmentationImageFilter.hxx"
