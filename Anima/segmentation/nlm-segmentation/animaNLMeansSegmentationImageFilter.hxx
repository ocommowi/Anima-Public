#pragma once
#include "animaNLMeansSegmentationImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkTimeProbe.h>

#include <animaNLMeansSegmentationPatchSearcher.h>
#include <animaMeanAndVarianceImagesFilter.h>

namespace anima
{

template <class PixelScalarType, class PixelOutputScalarType>
void
NLMeansSegmentationImageFilter <PixelScalarType,PixelOutputScalarType>
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
NLMeansSegmentationImageFilter <PixelScalarType,PixelOutputScalarType>
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

    typedef anima::NLMeansSegmentationPatchSearcher <InputImageType, DataImageType, OutputImageType> PatchSearcherType;

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
