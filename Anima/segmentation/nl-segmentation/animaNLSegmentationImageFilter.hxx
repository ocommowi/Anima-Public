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

    typedef itk::ImageRegionConstIterator <OutputImageType> SegmentationIteratorType;

    unsigned int maxSegmentationIndex = 0;
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

        SegmentationIteratorType segItr(m_DatabaseSegmentationImages[i],m_DatabaseSegmentationImages[i]->GetLargestPossibleRegion());
        while (!segItr.IsAtEnd())
        {
            if (segItr.Get() > maxSegmentationIndex)
                maxSegmentationIndex = segItr.Get();

            ++segItr;
        }
    }

    m_TemporaryOutputImage = VectorImageType::New();
    m_TemporaryOutputImage->Initialize();
    m_TemporaryOutputImage->SetRegions(this->GetInput()->GetLargestPossibleRegion());
    m_TemporaryOutputImage->SetSpacing (this->GetInput()->GetSpacing());
    m_TemporaryOutputImage->SetOrigin (this->GetInput()->GetOrigin());
    m_TemporaryOutputImage->SetDirection (this->GetInput()->GetDirection());
    m_TemporaryOutputImage->SetNumberOfComponentsPerPixel(maxSegmentationIndex + 1);
    m_TemporaryOutputImage->Allocate();

    typename VectorImageType::PixelType zeroPixel(maxSegmentationIndex + 1);
    zeroPixel.Fill(0);
    m_TemporaryOutputImage->FillBuffer(zeroPixel);

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
::SelectClosestAtlases(RegionType &patchNeighborhood, std::vector <unsigned int> &selectedAtlases)
{
    if ((m_NumberOfSelectedAtlases > m_DatabaseImages.size())||(m_NumberOfSelectedAtlases <= 0))
    {
        selectedAtlases.resize(m_DatabaseImages.size());
        for (unsigned int i = 0;i < m_DatabaseImages.size();++i)
            selectedAtlases[i] = i;

        return;
    }

    std::vector < std::pair <unsigned int, double> > ssdAtlases(m_DatabaseImages.size());
    typedef itk::ImageRegionConstIterator <InputImageType> InputIteratorType;
    typedef itk::ImageRegionConstIterator <DataImageType> DataIteratorType;

    InputIteratorType inItr(this->GetInput(), patchNeighborhood);
    for (unsigned int i = 0;i < m_DatabaseImages.size();++i)
    {
        inItr.GoToBegin();
        DataIteratorType dataItr(m_DatabaseImages[i],patchNeighborhood);

        double ssdValue = 0;
        while (!dataItr.IsAtEnd())
        {
            double inValue = inItr.Get();
            double dataValue = dataItr.Get();
            ssdValue += (inValue - dataValue) * (inValue - dataValue);

            ++dataItr;
            ++inItr;
        }

        ssdAtlases[i] = std::make_pair(i,ssdValue);
    }

    std::partial_sort(ssdAtlases.begin(),ssdAtlases.begin() + m_NumberOfSelectedAtlases, ssdAtlases.end(),pair_comparator());

    selectedAtlases.resize(m_NumberOfSelectedAtlases);
    for (unsigned int i = 0;i < m_NumberOfSelectedAtlases;++i)
        selectedAtlases[i] = ssdAtlases[i].first;
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
    typedef itk::ImageRegionIteratorWithIndex <VectorImageType> OutRegionIteratorType;
    typedef itk::ImageRegionConstIteratorWithIndex <MaskImageType> MaskRegionIteratorType;

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
    patchSearcher.SetIgnoreCenterPatches(false);

    std::vector <double> labelWeights;
    RegionType patchRegion;
    IndexType currentIndex;
    RegionType largestRegion = this->GetInput()->GetLargestPossibleRegion();
    std::vector <unsigned int> selectedAtlases;
    while (!maskIterator.IsAtEnd())
    {
        if (maskIterator.Get() == 0)
        {
            ++maskIterator;

            for (unsigned int k = 0;k < numSamplesDatabase;++k)
                ++databaseSegIterators[k];

            continue;
        }

        currentIndex = maskIterator.GetIndex();

        patchSearcher.ClearComparisonImages();
        patchSearcher.ClearSegmentationImages();
        patchSearcher.ClearComparisonMeanImages();
        patchSearcher.ClearComparisonVarImages();

        for (unsigned int i = 0;i < InputImageType::ImageDimension;++i)
        {
            int minIndex = std::max(largestRegion.GetIndex()[i],currentIndex[i] - m_PatchHalfSize - m_SearchNeighborhood);
            int maxIndex = std::min((int)(largestRegion.GetIndex()[i] + largestRegion.GetSize()[i] - 1),
                                    (int)(currentIndex[i] + m_PatchHalfSize + m_SearchNeighborhood));

            patchRegion.SetIndex(i,minIndex);
            patchRegion.SetSize(i,maxIndex - minIndex + 1);
        }

        this->SelectClosestAtlases(patchRegion, selectedAtlases);
        for (unsigned int k = 0;k < selectedAtlases.size();++k)
        {
            patchSearcher.AddComparisonImage(m_DatabaseImages[selectedAtlases[k]]);
            patchSearcher.AddSegmentationImage(m_DatabaseSegmentationImages[selectedAtlases[k]]);
            patchSearcher.AddComparisonMeanImage(m_DatabaseMeanImages[selectedAtlases[k]]);
            patchSearcher.AddComparisonVarImage(m_DatabaseVarImages[selectedAtlases[k]]);
        }

        patchSearcher.UpdateAtPosition(maskIterator.GetIndex());

        databaseSegSamples = patchSearcher.GetDatabaseSegmentationSamples();
        databaseWeights = patchSearcher.GetDatabaseWeights();

        // Add center pixels if no data selected
        if (databaseSegSamples.size() == 0)
        {
            for (unsigned int k = 0;k < numSamplesDatabase;++k)
            {
                databaseSegSamples.push_back(databaseSegIterators[k].Get());
                databaseWeights.push_back(1.0);
            }
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

        RegionType outRegion;
        RegionType computationRegion = this->GetComputationRegion();
        for (unsigned int i = 0;i < InputImageType::ImageDimension;++i)
        {
            int minIndex = std::max(computationRegion.GetIndex()[i],currentIndex[i] - m_PatchHalfSize);
            int maxIndex = std::min((int)(computationRegion.GetIndex()[i] + computationRegion.GetSize()[i] - 1), (int)(currentIndex[i] + m_PatchHalfSize));
            outRegion.SetIndex(i,minIndex);
            outRegion.SetSize(i,maxIndex - minIndex + 1);
        }

        m_LockTemporaryOutputImage.Lock();

        OutRegionIteratorType outItr(m_TemporaryOutputImage, outRegion);
        IndexType outIndex;
        VectorType outVector;
        while (!outItr.IsAtEnd())
        {
            double distCenter = 0;
            outIndex = outItr.GetIndex();

            for (unsigned int i = 0;i < InputImageType::ImageDimension;++i)
                distCenter += (outIndex[i] - currentIndex[i]) * (outIndex[i] - currentIndex[i]);

            outVector = outItr.Get();
            outVector[outSegValue] += std::exp(- distCenter / (m_PatchHalfSize * m_PatchHalfSize));

            outItr.Set(outVector);

            ++outItr;
        }

        m_LockTemporaryOutputImage.Unlock();

        ++maskIterator;

        for (unsigned int k = 0;k < numSamplesDatabase;++k)
            ++databaseSegIterators[k];
    }
}

template <class PixelScalarType, class PixelOutputScalarType>
void
NLSegmentationImageFilter <PixelScalarType,PixelOutputScalarType>
::AfterThreadedGenerateData()
{
    // Processes temporary output to finally get the true regularized output values
    VectorType outVector;
    typedef itk::ImageRegionConstIterator <VectorImageType> VectorIteratorType;
    typedef itk::ImageRegionIterator <OutputImageType> OutputIteratorType;
    typedef itk::ImageRegionConstIterator <MaskImageType> MaskIteratorType;

    OutputIteratorType outItr(this->GetOutput(),this->GetComputationRegion());
    MaskIteratorType maskItr(this->GetComputationMask(),this->GetComputationRegion());
    VectorIteratorType tempOutItr(m_TemporaryOutputImage,this->GetComputationRegion());

    while (!maskItr.IsAtEnd())
    {
        if (maskItr.Get() == 0)
        {
            ++outItr;
            ++maskItr;
            ++tempOutItr;

            continue;
        }

        outVector = tempOutItr.Get();
        unsigned int maxIndex = 0;
        double maxValue = outVector[0];

        for (unsigned int i = 1;i < outVector.GetSize();++i)
        {
            if (maxValue < outVector[i])
            {
                maxValue = outVector[i];
                maxIndex = i;
            }
        }

        outItr.Set(maxIndex);

        ++outItr;
        ++maskItr;
        ++tempOutItr;
    }
}

} //end namespace anima
