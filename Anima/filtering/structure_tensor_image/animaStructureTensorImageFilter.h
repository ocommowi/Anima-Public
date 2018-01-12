#pragma once

#include <iostream>
#include <itkImageToImageFilter.h>
#include <itkImage.h>
#include <itkVectorImage.h>

namespace anima
{

/**
 * @brief Compute the structure tensor image from blocks in the image
 * using a gradient image as an input
 */
template <typename TInputPixelType, typename TOutputScalarType, unsigned int Dimension>
class StructureTensorImageFilter :
public itk::ImageToImageFilter< itk::Image <TInputPixelType, Dimension> ,
        itk::VectorImage <TOutputScalarType, Dimension> >
{
public:
    typedef StructureTensorImageFilter Self;
    typedef itk::Image <TInputPixelType, Dimension> InputImageType;
    typedef itk::VectorImage <TOutputScalarType, Dimension> OutputImageType;
    typedef itk::ImageToImageFilter <InputImageType, OutputImageType> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    itkNewMacro(Self)
    itkTypeMacro(StructureTensorImageFilter, itk::ImageToImageFilter)

    typedef typename InputImageType::IndexType IndexType;
    typedef typename InputImageType::RegionType RegionType;
    typedef typename InputImageType::PointType PointType;
    typedef typename InputImageType::PixelType InputPixelType;
    typedef typename OutputImageType::PixelType OutputPixelType;

    typedef typename InputImageType::Pointer InputImagePointer;
    typedef typename OutputImageType::Pointer OutputImagePointer;

    typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

    itkSetMacro(Neighborhood, unsigned int)
    itkSetMacro(Normalize, bool)

protected:
    StructureTensorImageFilter()
    {
        m_Neighborhood = 2;
        m_Normalize = false;
    }

    virtual ~StructureTensorImageFilter() {}

    void BeforeThreadedGenerateData() ITK_OVERRIDE;
    void GenerateOutputInformation() ITK_OVERRIDE;
    void ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, itk::ThreadIdType threadId) ITK_OVERRIDE;

private:
    StructureTensorImageFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    //! Neighborhood from which to compute structure tensor
    unsigned int m_Neighborhood;

    bool m_Normalize;
};

} // end namespace anima

#include "animaStructureTensorImageFilter.hxx"
