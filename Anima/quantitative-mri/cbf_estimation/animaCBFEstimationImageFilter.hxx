#pragma once
#include "animaCBFEstimationImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>

namespace anima
{

template <class InputPixelType, class OutputPixelType>
void
CBFEstimationImageFilter <InputPixelType,OutputPixelType>
::BeforeThreadedGenerateData ()
{
    this->Superclass::BeforeThreadedGenerateData();

    if (!m_M0Image)
    {
        m_M0Image = InputImageType::New();
        m_M0Image->Initialize();
        m_M0Image->SetRegions(this->GetInput(0)->GetLargestPossibleRegion());
        m_M0Image->SetSpacing (this->GetInput(0)->GetSpacing());
        m_M0Image->SetOrigin (this->GetInput(0)->GetOrigin());
        m_M0Image->SetDirection (this->GetInput(0)->GetDirection());
        m_M0Image->Allocate();
        m_M0Image->FillBuffer(m_M0ConstantValue);
    }
}

template <class InputPixelType, class OutputPixelType>
void
CBFEstimationImageFilter <InputPixelType,OutputPixelType>
::ThreadedGenerateData (const OutputImageRegionType &outputRegionForThread, itk::ThreadIdType threadId)
{
    typedef itk::ImageRegionConstIteratorWithIndex <InputImageType> InputImageIteratorType;
    typedef itk::ImageRegionConstIterator <MaskImageType> MaskIteratorType;
    typedef itk::ImageRegionIterator <OutputImageType> OutputImageIteratorType;
    
    InputImageIteratorType inputItr(this->GetInput(0), outputRegionForThread);
    InputImageIteratorType m0Itr(m_M0Image, outputRegionForThread);
    MaskIteratorType maskItr(this->GetComputationMask(),outputRegionForThread);
    OutputImageIteratorType outItr(this->GetOutput(0),outputRegionForThread);
    
    IndexType currentIndex;
    double constantValue = 6.0e6 * m_LambdaParameter / (m_AlphaParameter * m_BloodT1 * (1.0 - std::exp(- m_LabelDuration / m_BloodT1)));
    while (!inputItr.IsAtEnd())
    {
        if (maskItr.Get() == 0)
        {
            ++maskItr;
            ++m0Itr;
            ++inputItr;
            ++outItr;

            continue;
        }

        currentIndex = inputItr.GetIndex();
        double currentPLD = currentIndex[InputImageType::ImageDimension - 1] * m_SliceDelay + m_BasePostLabelingDelay;

        double cbfValue = constantValue * inputItr.Get() / (m0Itr.Get() * std::exp(-currentPLD / m_BloodT1));

        outItr.Set(cbfValue);
        
        ++maskItr;
        ++m0Itr;
        ++inputItr;
        ++outItr;
    }
}

} //end of namespace anima

