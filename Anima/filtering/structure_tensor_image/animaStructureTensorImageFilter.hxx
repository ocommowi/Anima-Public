#pragma once
#include "animaStructureTensorImageFilter.h"

#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>

#include <animaBaseTensorTools.h>
#include <itkSymmetricEigenAnalysis.h>

namespace anima
{

template <typename TInputPixelType, typename TOutputScalarType, unsigned int Dimension>
void
StructureTensorImageFilter <TInputPixelType, TOutputScalarType, Dimension>
::GenerateOutputInformation()
{
    // Override the method in itkImageSource, so we can set the vector length of
    // the output itk::VectorImage

    this->Superclass::GenerateOutputInformation();

    unsigned int vectorLength = Dimension * (Dimension + 1) / 2;
    OutputImageType *output = this->GetOutput();
    output->SetVectorLength(vectorLength);
}

template <typename TInputPixelType, typename TOutputScalarType, unsigned int Dimension>
void
StructureTensorImageFilter <TInputPixelType, TOutputScalarType, Dimension>
::BeforeThreadedGenerateData()
{
    this->Superclass::BeforeThreadedGenerateData();

    if (this->GetInput()->GetNumberOfComponentsPerPixel() != Dimension)
        itkExceptionMacro("Expected N components per pixel, ND gradient image as an input");
}

template <typename TInputPixelType, typename TOutputScalarType, unsigned int Dimension>
void
StructureTensorImageFilter <TInputPixelType, TOutputScalarType, Dimension>
::ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, itk::ThreadIdType threadId)
{
    typedef itk::ImageRegionConstIteratorWithIndex <InputImageType> InputIteratorType;
    typedef itk::ImageRegionConstIterator <InputImageType> InternalInputIteratorType;
    typedef itk::ImageRegionIterator <OutputImageType> OutputIteratorType;

    InputIteratorType inputItr(this->GetInput(),outputRegionForThread);
    OutputIteratorType outputItr(this->GetOutput(),outputRegionForThread);

    IndexType currentIndex;
    vnl_matrix <double> structureTensor(Dimension,Dimension);

    typedef itk::SymmetricEigenAnalysis < vnl_matrix <double>, vnl_diag_matrix<double>, vnl_matrix <double> > EigenAnalysisType;

    EigenAnalysisType eigen(Dimension);
    vnl_matrix <double> eigVecs(Dimension,Dimension);
    vnl_diag_matrix <double> eigVals(Dimension);

    RegionType inputRegion;
    RegionType largestRegion = this->GetInput()->GetLargestPossibleRegion();
    InputPixelType pixelValue;
    OutputPixelType outputValue;

    while (!inputItr.IsAtEnd())
    {
        currentIndex = inputItr.GetIndex();

        // Build the explored region
        for (unsigned int i = 0;i < Dimension;++i)
        {
            int largestIndex = largestRegion.GetIndex()[i];
            int testedIndex = currentIndex[i] - m_Neighborhood;
            unsigned int baseIndex = std::max(largestIndex,testedIndex);
            unsigned int maxValue = largestIndex + largestRegion.GetSize()[i] - 1;
            unsigned int upperIndex = std::min(maxValue,(unsigned int)(currentIndex[i] + m_Neighborhood));

            inputRegion.SetIndex(i,baseIndex);
            inputRegion.SetSize(i,upperIndex - baseIndex + 1);
        }


        structureTensor.fill(0.0);

        InternalInputIteratorType internalItr(this->GetInput(),inputRegion);
        while (!internalItr.IsAtEnd())
        {
            pixelValue = internalItr.Get();
            for (unsigned int i = 0;i < Dimension;++i)
            {
                for (unsigned int j = i;j < Dimension;++j)
                    structureTensor(i,j) += pixelValue[i] * pixelValue[j];
            }

            ++internalItr;
        }

        for (unsigned int i = 0;i < Dimension;++i)
        {
            for (unsigned int j = i + 1;j < Dimension;++j)
                structureTensor(j,i) = structureTensor(i,j);
        }

        structureTensor /= inputRegion.GetNumberOfPixels();

        if (m_Normalize)
        {
            eigen.ComputeEigenValuesAndVectors(structureTensor,eigVals,eigVecs);

            if (eigVals[0] != 0)
            {
                double minEig = eigVals[0];
                for (unsigned int i = 0;i < Dimension;++i)
                    eigVals[i] = std::sqrt(minEig / eigVals[i]);
            }
            else
            {
                for (unsigned int i = 0;i < Dimension;++i)
                    eigVals[i] = 1.0;
            }

            double volumeTensor = eigVals[0];
            for (unsigned int i = 1;i < Dimension;++i)
                volumeTensor *= eigVals[i];

            volumeTensor = std::pow(volumeTensor, 1.0 / 3);

            double constMinScale = 4.0 / (2.0 * m_Neighborhood + 1.0);
            double constMaxScale = 1.0 / constMinScale;
            for (unsigned int i = 0;i < Dimension;++i)
                eigVals[i] = std::min(constMaxScale,std::max(constMinScale,eigVals[i] / volumeTensor));

            anima::RecomposeTensor(eigVals,eigVecs,structureTensor);
        }

        anima::GetVectorRepresentation(structureTensor,outputValue);
        outputItr.Set(outputValue);

        ++inputItr;
        ++outputItr;
    }
}

} //end of namespace anima
