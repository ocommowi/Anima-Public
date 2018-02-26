#pragma once
#include "animaNLSegmentationImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkAddImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkThresholdLabelerImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkGrayscaleDilateImageFilter.h>

#include <animaNLSegmentationPatchSearcher.h>
#include <animaMeanAndVarianceImagesFilter.h>

namespace anima
{

template <class PixelScalarType, class PixelOutputScalarType>
void
NLSegmentationImageFilter <PixelScalarType,PixelOutputScalarType>
::BeforeThreadedGenerateData()
{
    Superclass::BeforeThreadedGenerateData();

    // Checking consistency of the data and parameters

    unsigned int nbInputs = this->GetNumberOfIndexedInputs();
    if (nbInputs <= 0)
        itkExceptionMacro("No input available...");

    if (m_DatabaseImages.size() != m_DatabaseSegmentationImages.size())
        itkExceptionMacro("There should be the same number of input segmentations and database images...")

    typedef anima::MeanAndVarianceImagesFilter<InputImageType, DataImageType> MeanVarianceFilterType;
    typename InputImageType::SizeType radius;
    for (unsigned int j = 0;j < InputImageType::ImageDimension;++j)
        radius[j] = m_PatchHalfSize;

    for (unsigned int i = 0;i < m_DatabaseImages.size();++i)
    {
        typename MeanVarianceFilterType::Pointer filter = MeanVarianceFilterType::New();
        filter->SetInput(m_DatabaseImages[i]);

        filter->SetRadius(radius);
        filter->SetNumberOfThreads(this->GetNumberOfThreads());
        filter->Update();

        m_DatabaseMeanImages.push_back(filter->GetMeanImage());
        m_DatabaseMeanImages[i]->DisconnectPipeline();

        m_DatabaseVarImages.push_back(filter->GetVarImage());
        m_DatabaseVarImages[i]->DisconnectPipeline();
    }

    typename MeanVarianceFilterType::Pointer filter = MeanVarianceFilterType::New();
    filter->SetInput(this->GetInput());

    filter->SetRadius(radius);
    filter->SetNumberOfThreads(this->GetNumberOfThreads());
    filter->Update();

    m_ReferenceMeanImage = filter->GetMeanImage();
    m_ReferenceMeanImage->DisconnectPipeline();

    m_ReferenceVarImage = filter->GetVarImage();
    m_ReferenceVarImage->DisconnectPipeline();
}

template <class PixelScalarType, class PixelOutputScalarType>
void
NLSegmentationImageFilter <PixelScalarType,PixelOutputScalarType>
::CheckComputationMask()
{
    if (this->GetComputationMask())
        return;

    typedef itk::AddImageFilter <OutputImageType, OutputImageType, OutputImageType> AddFilterType;

    OutputImagePointer outImg = OutputImageType::New();
    outImg->Initialize();
    outImg->SetRegions(this->GetInput()->GetLargestPossibleRegion());
    outImg->SetSpacing (this->GetInput()->GetSpacing());
    outImg->SetOrigin (this->GetInput()->GetOrigin());
    outImg->SetDirection (this->GetInput()->GetDirection());
    outImg->Allocate();

    outImg->FillBuffer(0);

    for (unsigned int i = 0;i < m_DatabaseSegmentationImages.size();++i)
    {
        typename AddFilterType::Pointer addFilter = AddFilterType::New();
        addFilter->SetInput1(outImg);
        addFilter->SetInput2(m_DatabaseSegmentationImages[i]);
        addFilter->SetNumberOfThreads(this->GetNumberOfThreads());

        addFilter->Update();
        outImg = addFilter->GetOutput();
        outImg->DisconnectPipeline();
    }

    typedef itk::ThresholdImageFilter <OutputImageType> ThresholdFilterType;
    typename ThresholdFilterType::Pointer thrFilter = ThresholdFilterType::New();
    thrFilter->SetInput(outImg);
    thrFilter->SetOutsideValue(0);
    thrFilter->SetNumberOfThreads(this->GetNumberOfThreads());

    thrFilter->Update();

    typedef itk::ThresholdLabelerImageFilter <OutputImageType,MaskImageType> LabelerFilterType;
    typename LabelerFilterType::Pointer labelFilter = LabelerFilterType::New();
    labelFilter->SetInput(thrFilter->GetOutput());

    typename LabelerFilterType::RealThresholdVector thrVals;
    thrVals.push_back(0);

    labelFilter->SetRealThresholds(thrVals);
    labelFilter->SetNumberOfThreads(this->GetNumberOfThreads());

    labelFilter->Update();

    typename MaskImageType::Pointer maskImg = labelFilter->GetOutput();
    maskImg->DisconnectPipeline();

    typedef itk::BinaryBallStructuringElement <unsigned short, 3> BallElementType;
    typedef itk::GrayscaleDilateImageFilter <MaskImageType,MaskImageType,BallElementType> DilateFilterType;

    typename DilateFilterType::Pointer dilateFilter = DilateFilterType::New();
    dilateFilter->SetInput(maskImg);
    dilateFilter->SetNumberOfThreads(this->GetNumberOfThreads());

    BallElementType tmpBall;
    BallElementType::SizeType ballSize;

    for (unsigned int i = 0;i < InputImageType::ImageDimension;++i)
        ballSize[i] = m_SearchNeighborhood + 1;

    tmpBall.SetRadius(ballSize);
    tmpBall.CreateStructuringElement();

    dilateFilter->SetKernel(tmpBall);
    dilateFilter->Update();

    maskImg = dilateFilter->GetOutput();
    maskImg->DisconnectPipeline();

    this->SetComputationMask(maskImg);
}

template <class PixelScalarType, class PixelOutputScalarType>
void
NLSegmentationImageFilter <PixelScalarType,PixelOutputScalarType>
::ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, itk::ThreadIdType threadId)
{
    typedef itk::ImageRegionConstIterator <OutputImageType> SegmentationIteratorType;
    typedef itk::ImageRegionIterator <OutputImageType> OutRegionIteratorType;
    typedef itk::ImageRegionConstIteratorWithIndex <MaskImageType> MaskRegionIteratorType;

    OutRegionIteratorType outIterator(this->GetOutput(0), outputRegionForThread);
    MaskRegionIteratorType maskIterator (this->GetComputationMask(), outputRegionForThread);

    unsigned int numSamplesDatabase = m_DatabaseSegmentationImages.size();
    std::vector <SegmentationIteratorType> databaseSegIterators(numSamplesDatabase);
    for (unsigned int k = 0;k < numSamplesDatabase;++k)
        databaseSegIterators[k] = SegmentationIteratorType(m_DatabaseSegmentationImages[k],outputRegionForThread);

    std::vector <PixelOutputScalarType> databaseSegSamples;
    std::vector <double> databaseWeights;

    int maxAbsDisp = (int)std::floor((double)(m_SearchNeighborhood / m_SearchStepSize)) * m_SearchStepSize;

    typedef anima::NLSegmentationPatchSearcher <InputImageType, DataImageType, OutputImageType> PatchSearcherType;

    InputImageType *input = const_cast <InputImageType *> (this->GetInput());

    PatchSearcherType patchSearcher;
    patchSearcher.SetPatchHalfSize(m_PatchHalfSize);
    patchSearcher.SetSearchStepSize(m_SearchStepSize);
    patchSearcher.SetMaxAbsDisp(maxAbsDisp);
    patchSearcher.SetInputImage(input);
    patchSearcher.SetWeightThreshold(m_WeightThreshold);
    patchSearcher.SetThreshold(m_Threshold);
    patchSearcher.SetMeanImage(m_ReferenceMeanImage);
    patchSearcher.SetVarImage(m_ReferenceVarImage);

    for (unsigned int k = 0;k < numSamplesDatabase;++k)
    {
        patchSearcher.AddComparisonImage(m_DatabaseImages[k]);
        patchSearcher.AddSegmentationImage(m_DatabaseSegmentationImages[k]);
        patchSearcher.AddComparisonMeanImage(m_DatabaseMeanImages[k]);
        patchSearcher.AddComparisonVarImage(m_DatabaseVarImages[k]);
    }

    std::vector <double> labelWeights;
    while (!outIterator.IsAtEnd())
    {
        if (maskIterator.Get() == 0)
        {
            outIterator.Set(0);
            ++outIterator;
            ++maskIterator;

            for (unsigned int k = 0;k < numSamplesDatabase;++k)
                ++databaseSegIterators[k];

            continue;
        }

        patchSearcher.UpdateAtPosition(maskIterator.GetIndex());

        databaseSegSamples = patchSearcher.GetDatabaseSegmentationSamples();
        databaseWeights = patchSearcher.GetDatabaseWeights();

        double maxSamplesWeight = 0;
        for (unsigned int k = 0;k < databaseWeights.size();++k)
        {
            if (maxSamplesWeight < databaseWeights[k])
                maxSamplesWeight = databaseWeights[k];
        }

        // Add center pixels
        if (maxSamplesWeight == 0)
            maxSamplesWeight = 1.0;

        for (unsigned int k = 0;k < numSamplesDatabase;++k)
        {
            databaseSegSamples.push_back(databaseSegIterators[k].Get());
            databaseWeights.push_back(maxSamplesWeight);
        }

        PixelOutputScalarType outSegValue = 0;

        unsigned int maxLabel = 0;
        for (unsigned int k = 0;k < databaseSegSamples.size();++k)
        {
            if (databaseSegSamples[k] > maxLabel)
                maxLabel = databaseSegSamples[k];
        }

        labelWeights.resize(maxLabel + 1);
        std::fill(labelWeights.begin(),labelWeights.end(),0.0);

        for (unsigned int k = 0;k < databaseSegSamples.size();++k)
            labelWeights[databaseSegSamples[k]] += databaseWeights[k];

        double maxWeight = labelWeights[0];
        for (unsigned int k = 1;k <= maxLabel;++k)
        {
            if (maxWeight < labelWeights[k])
            {
                maxWeight = labelWeights[k];
                outSegValue = k;
            }
        }

        outIterator.Set(outSegValue);

        ++outIterator;
        ++maskIterator;

        for (unsigned int k = 0;k < numSamplesDatabase;++k)
            ++databaseSegIterators[k];
    }
}

} //end namespace anima
