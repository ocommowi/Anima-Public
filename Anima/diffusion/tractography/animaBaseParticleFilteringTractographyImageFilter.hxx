#include "animaBaseProbabilisticTractographyImageFilter.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkExtractImageFilter.h>
#include <itkImageMomentsCalculator.h>

#include <animaVectorOperations.h>
#include <animaLogarithmFunctions.h>

#include <vnl/algo/vnl_matrix_inverse.h>

#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>

#include <animaKMeansFilter.h>

#include <ctime>

namespace anima
{

template <class TInputModelImageType>
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::BaseProbabilisticTractographyImageFilter()
{
    m_PointsToProcess.clear();

    m_NumberOfFibersPerPixel = 1;
    m_NumberOfParticles = 1000;
    m_MinimalNumberOfParticlesPerClass = 10;

    m_ResamplingThreshold = 0.8;

    m_StepProgression = 1.0;

    m_KappaOfPriorDistribution = 30.0;

    m_MinLengthFiber = 10.0;
    m_MaxLengthFiber = 150.0;

    m_PositionDistanceFuseThreshold = 0.5;
    m_KappaSplitThreshold = 30.0;

    m_ClusterDistance = 0;

    m_ComputeLocalColors = true;
    m_MAPMergeFibers = true;

    m_Generators.clear();

    m_HighestProcessedSeed = 0;
    m_ProgressReport = 0;
}

template <class TInputModelImageType>
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::~BaseProbabilisticTractographyImageFilter()
{
    if (m_ProgressReport)
        delete m_ProgressReport;
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::AddGradientDirection(unsigned int i, Vector3DType &grad)
{
    if (i == m_DiffusionGradients.size())
        m_DiffusionGradients.push_back(grad);
    else if (i > m_DiffusionGradients.size())
        std::cerr << "Trying to add a direction not contiguous... Add directions contiguously (0,1,2,3,...)..." << std::endl;
    else
        m_DiffusionGradients[i] = grad;
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::Update()
{
    this->PrepareTractography();
    m_Output = vtkPolyData::New();

    if (m_ProgressReport)
        delete m_ProgressReport;

    unsigned int stepData = std::min((int)m_PointsToProcess.size(),100);
    if (stepData == 0)
        stepData = 1;

    unsigned int numSteps = std::floor(m_PointsToProcess.size() / (double)stepData);
    if (m_PointsToProcess.size() % stepData != 0)
        numSteps++;

    m_ProgressReport = new itk::ProgressReporter(this,0,numSteps);

    FiberProcessVectorType resultFibers;
    ListType resultWeights;

    trackerArguments tmpStr;
    tmpStr.trackerPtr = this;
    tmpStr.resultFibersFromThreads.resize(this->GetNumberOfWorkUnits());
    tmpStr.resultWeightsFromThreads.resize(this->GetNumberOfWorkUnits());

    for (unsigned int i = 0;i < this->GetNumberOfWorkUnits();++i)
    {
        tmpStr.resultFibersFromThreads[i] = resultFibers;
        tmpStr.resultWeightsFromThreads[i] = resultWeights;
    }

    this->GetMultiThreader()->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
    this->GetMultiThreader()->SetSingleMethod(this->ThreadTracker,&tmpStr);
    this->GetMultiThreader()->SingleMethodExecute();

    for (unsigned int j = 0;j < this->GetNumberOfWorkUnits();++j)
    {
        resultFibers.insert(resultFibers.end(),tmpStr.resultFibersFromThreads[j].begin(),tmpStr.resultFibersFromThreads[j].end());
        resultWeights.insert(resultWeights.end(),tmpStr.resultWeightsFromThreads[j].begin(),tmpStr.resultWeightsFromThreads[j].end());
    }

    std::cout << "\nKept " << resultFibers.size() << " fibers after filtering" << std::endl;
    this->createVTKOutput(resultFibers, resultWeights);
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::PrepareTractography()
{
    if (!m_B0Image)
        itkExceptionMacro("No B0 image, required");

    if (!m_NoiseImage)
        itkExceptionMacro("No sigma square noise image, required");

    m_B0Interpolator = ScalarInterpolatorType::New();
    m_B0Interpolator->SetInputImage(m_B0Image);

    m_NoiseInterpolator = ScalarInterpolatorType::New();
    m_NoiseInterpolator->SetInputImage(m_NoiseImage);

    // Initialize random generator
    m_Generators.resize(this->GetNumberOfWorkUnits());

    std::mt19937 motherGenerator(time(0));

    for (unsigned int i = 0;i < this->GetNumberOfWorkUnits();++i)
        m_Generators[i] = std::mt19937(motherGenerator());

    bool is2d = m_InputModelImage->GetLargestPossibleRegion().GetSize()[2] == 1;
    typedef itk::ImageRegionIteratorWithIndex <MaskImageType> MaskImageIteratorType;

    MaskImageIteratorType maskItr(m_SeedMask, m_InputModelImage->GetLargestPossibleRegion());
    m_PointsToProcess.clear();

    IndexType tmpIndex;
    PointType tmpPoint;
    ContinuousIndexType realIndex;

    m_FilteringValues.clear();
    double startN = -0.5 + 1.0 / (2.0 * m_NumberOfFibersPerPixel);
    double stepN = 1.0 / m_NumberOfFibersPerPixel;
    FiberType tmpFiber(1);

    if (m_FilterMask)
    {
        MaskImageIteratorType filterItr(m_FilterMask, m_InputModelImage->GetLargestPossibleRegion());
        while (!filterItr.IsAtEnd())
        {
            if (filterItr.Get() == 0)
            {
                ++filterItr;
                continue;
            }

            bool isAlreadyIn = false;
            for (unsigned int i = 0;i < m_FilteringValues.size();++i)
            {
                if (m_FilteringValues[i] == filterItr.Get())
                {
                    isAlreadyIn = true;
                    break;
                }
            }

            if (!isAlreadyIn)
                m_FilteringValues.push_back(filterItr.Get());

            ++filterItr;
        }
    }

    while (!maskItr.IsAtEnd())
    {
        if (maskItr.Get() == 0)
        {
            ++maskItr;
            continue;
        }

        tmpIndex = maskItr.GetIndex();

        if (is2d)
        {
            realIndex[2] = tmpIndex[2];
            for (unsigned int j = 0;j < m_NumberOfFibersPerPixel;++j)
            {
                realIndex[1] = tmpIndex[1] + startN + j * stepN;
                for (unsigned int i = 0;i < m_NumberOfFibersPerPixel;++i)
                {
                    realIndex[0] = tmpIndex[0] + startN + i * stepN;
                    m_SeedMask->TransformContinuousIndexToPhysicalPoint(realIndex,tmpPoint);
                    tmpFiber[0] = tmpPoint;
                    m_PointsToProcess.push_back(tmpFiber);
                }
            }
        }
        else
        {
            for (unsigned int k = 0;k < m_NumberOfFibersPerPixel;++k)
            {
                realIndex[2] = tmpIndex[2] + startN + k * stepN;
                for (unsigned int j = 0;j < m_NumberOfFibersPerPixel;++j)
                {
                    realIndex[1] = tmpIndex[1] + startN + j * stepN;
                    for (unsigned int i = 0;i < m_NumberOfFibersPerPixel;++i)
                    {
                        realIndex[0] = tmpIndex[0] + startN + i * stepN;

                        m_SeedMask->TransformContinuousIndexToPhysicalPoint(realIndex,tmpPoint);
                        tmpFiber[0] = tmpPoint;
                        m_PointsToProcess.push_back(tmpFiber);
                    }
                }
            }
        }

        ++maskItr;
    }

    std::cout << "Generated " << m_PointsToProcess.size() << " seed points from ROI mask" << std::endl;
}

template <class TInputModelImageType>
typename BaseProbabilisticTractographyImageFilter <TInputModelImageType>::InterpolatorType *
BaseProbabilisticTractographyImageFilter <TInputModelImageType>::GetModelInterpolator()
{
    typedef itk::LinearInterpolateImageFunction <InputModelImageType> InternalInterpolatorType;

    typename InternalInterpolatorType::Pointer outInterpolator = InternalInterpolatorType::New();
    outInterpolator->SetInputImage(m_InputModelImage);

    outInterpolator->Register();
    return outInterpolator;
}

template <class TInputModelImageType>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::ThreadTracker(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;

    trackerArguments *tmpArg = (trackerArguments *)threadArgs->UserData;
    tmpArg->trackerPtr->ThreadTrack(nbThread,tmpArg->resultFibersFromThreads[nbThread],tmpArg->resultWeightsFromThreads[nbThread]);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::ThreadTrack(unsigned int numThread, FiberProcessVectorType &resultFibers,
              ListType &resultWeights)
{
    bool continueLoop = true;
    unsigned int highestToleratedSeedIndex = m_PointsToProcess.size();

    unsigned int stepData = std::min((int)m_PointsToProcess.size(),100);
    if (stepData == 0)
        stepData = 1;

    while (continueLoop)
    {
        m_LockHighestProcessedSeed.lock();

        if (m_HighestProcessedSeed >= highestToleratedSeedIndex)
        {
            m_LockHighestProcessedSeed.unlock();
            continueLoop = false;
            continue;
        }

        unsigned int startPoint = m_HighestProcessedSeed;
        unsigned int endPoint = m_HighestProcessedSeed + stepData;
        if (endPoint > highestToleratedSeedIndex)
            endPoint = highestToleratedSeedIndex;

        m_HighestProcessedSeed = endPoint;

        m_LockHighestProcessedSeed.unlock();

        this->ThreadedTrackComputer(numThread,resultFibers,resultWeights,startPoint,endPoint);

        m_LockHighestProcessedSeed.lock();
        m_ProgressReport->CompletedPixel();
        m_LockHighestProcessedSeed.unlock();
    }
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::ThreadedTrackComputer(unsigned int numThread, FiberProcessVectorType &resultFibers,
                        ListType &resultWeights, unsigned int startSeedIndex,
                        unsigned int endSeedIndex)
{
    InterpolatorPointer modelInterpolator = this->GetModelInterpolator();
    FiberProcessVectorType tmpFibers;
    ListType tmpWeights;
    ContinuousIndexType startIndex;

    for (unsigned int i = startSeedIndex;i < endSeedIndex;++i)
    {
        m_SeedMask->TransformPhysicalPointToContinuousIndex(m_PointsToProcess[i][0],startIndex);

        tmpFibers = this->ComputeFiber(m_PointsToProcess[i], modelInterpolator, numThread, tmpWeights);

        tmpFibers = this->FilterOutputFibers(tmpFibers, tmpWeights);

        for (unsigned int j = 0;j < tmpFibers.size();++j)
        {
            if (tmpFibers[j].size() > m_MinLengthFiber / m_StepProgression)
            {
                resultFibers.push_back(tmpFibers[j]);
                resultWeights.push_back(tmpWeights[j]);
            }
        }
    }
}

template <class TInputModelImageType>
typename BaseProbabilisticTractographyImageFilter <TInputModelImageType>::FiberProcessVectorType
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::FilterOutputFibers(FiberProcessVectorType &fibers, ListType &weights)
{
    FiberProcessVectorType resVal;
    ListType tmpWeights = weights;
    weights.clear();

    if ((m_FilteringValues.size() > 0)||(m_ForbiddenMask))
    {
        MembershipType touchingLabels;
        IndexType tmpIndex;
        PointType tmpPoint;

        for (unsigned int i = 0;i < fibers.size();++i)
        {
            touchingLabels.clear();
            bool forbiddenTouched = false;

            for (unsigned int j = 0;j < fibers[i].size();++j)
            {
                tmpPoint = fibers[i][j];
                m_SeedMask->TransformPhysicalPointToIndex(tmpPoint,tmpIndex);

                unsigned int maskValue = 0;
                unsigned int forbiddenMaskValue = 0;

                if (m_FilterMask)
                    maskValue = m_FilterMask->GetPixel(tmpIndex);

                if (m_ForbiddenMask)
                    forbiddenMaskValue = m_ForbiddenMask->GetPixel(tmpIndex);

                if (forbiddenMaskValue != 0)
                {
                    forbiddenTouched = true;
                    break;
                }

                if (maskValue != 0)
                {
                    bool alreadyIn = false;
                    for (unsigned int k = 0;k < touchingLabels.size();++k)
                    {
                        if (maskValue == touchingLabels[k])
                        {
                            alreadyIn = true;
                            break;
                        }
                    }

                    if (!alreadyIn)
                        touchingLabels.push_back(maskValue);
                }
            }

            if (forbiddenTouched)
                continue;

            if (touchingLabels.size() == m_FilteringValues.size())
            {
                resVal.push_back(fibers[i]);
                weights.push_back(tmpWeights[i]);
            }
        }
    }
    else
    {
        resVal = fibers;
        weights = tmpWeights;
    }

    return resVal;
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::createVTKOutput(FiberProcessVectorType &filteredFibers, ListType &filteredWeights)
{
    m_Output = vtkPolyData::New();
    m_Output->Initialize();
    m_Output->Allocate();

    vtkSmartPointer <vtkPoints> myPoints = vtkPoints::New();
    vtkSmartPointer <vtkDoubleArray> weights = vtkDoubleArray::New();
    weights->SetNumberOfComponents(1);
    weights->SetName("Fiber weights");

    for (unsigned int i = 0;i < filteredFibers.size();++i)
    {
        unsigned int npts = filteredFibers[i].size();
        vtkIdType* ids = new vtkIdType[npts];

        for (unsigned int j = 0;j < npts;++j)
        {
            ids[j] = myPoints->InsertNextPoint(filteredFibers[i][j][0],filteredFibers[i][j][1],filteredFibers[i][j][2]);
            weights->InsertNextValue(std::exp(filteredWeights[i]));
        }

        m_Output->InsertNextCell (VTK_POLY_LINE, npts, ids);
        delete[] ids;
    }

    m_Output->SetPoints(myPoints);
    if (m_ComputeLocalColors)
        this->ComputeAdditionalScalarMaps();

    // Add particle weights to data
    m_Output->GetPointData()->AddArray(weights);
}

template <class TInputModelImageType>
typename BaseProbabilisticTractographyImageFilter <TInputModelImageType>::FiberProcessVectorType
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::ComputeFiber(FiberType &fiber, InterpolatorPointer &modelInterpolator,
               unsigned int numThread, ListType &resultWeights)
{
    unsigned int numberOfClasses = 1;

    FiberWorkType fiberComputationData;
    fiberComputationData.fiberParticles.resize(m_NumberOfParticles);
    for (unsigned int i = 0;i < m_NumberOfParticles;++i)
        fiberComputationData.fiberParticles[i] = fiber;

    fiberComputationData.logParticleWeights = ListType(m_NumberOfParticles, - std::log(m_NumberOfParticles));
    fiberComputationData.logNormalizedParticleWeights = ListType(m_NumberOfParticles, - std::log(m_NumberOfParticles));
    fiberComputationData.previousUpdateLogWeights = ListType(m_NumberOfParticles, - std::log(m_NumberOfParticles));
    fiberComputationData.stoppedParticles = std::vector <bool> (m_NumberOfParticles,false);
    fiberComputationData.classSizes = MembershipType(numberOfClasses,m_NumberOfParticles);
    fiberComputationData.logClassWeights = ListType(numberOfClasses, - std::log(numberOfClasses));
    fiberComputationData.fiberNumberOfPoints = std::vector <unsigned int> (m_NumberOfParticles, 1);
    //We need membership vectors in each direction
    fiberComputationData.classMemberships = MembershipType(m_NumberOfParticles,0);
    fiberComputationData.reverseClassMemberships.resize(numberOfClasses);
    fiberComputationData.reverseClassMemberships[0].resize(m_NumberOfParticles);
    for (unsigned int j = 0;j < m_NumberOfParticles;++j)
        fiberComputationData.reverseClassMemberships[0][j] = j;

    ListType logWeightSums(numberOfClasses,0);
    DirectionVectorType previousDirections(m_NumberOfParticles);

    // Data structures for resampling
    DirectionVectorType previousDirectionsCopy;

    VectorType modelValue(m_ModelDimension);
    PointType currentPoint;
    ContinuousIndexType currentIndex;

    // First check: is model right at start point?
    currentPoint = fiber.back();
    m_SeedMask->TransformPhysicalPointToContinuousIndex(currentPoint,currentIndex);
    modelValue.Fill(0.0);
    double estimatedNoiseValue = 20.0;
    this->ComputeModelValue(modelInterpolator, currentIndex, modelValue);
    double estimatedB0Value = m_B0Interpolator->EvaluateAtContinuousIndex(currentIndex);
    estimatedNoiseValue = m_NoiseInterpolator->EvaluateAtContinuousIndex(currentIndex);

    if (!this->CheckModelProperties(estimatedB0Value,estimatedNoiseValue,modelValue,numThread))
        return fiberComputationData.fiberParticles;

    this->InitializeFirstIterationFromModel(modelValue,numThread,previousDirections);

    // Perform the forward progression
    bool stopLoop = false;
    unsigned int maxIterations = m_MaxLengthFiber / m_StepProgression + 1;

    while (!stopLoop)
    {
        logWeightSums.resize(numberOfClasses);
        std::fill(logWeightSums.begin(),logWeightSums.end(),0.0);

        this->ProgressParticles(fiberComputationData,modelInterpolator,previousDirections,numThread);

        // Update weights
        this->UpdateWeightsFromCurrentData(fiberComputationData,logWeightSums);

        // Resampling if necessary
        this->CheckAndPerformOccasionalResampling(fiberComputationData,previousDirections,previousDirectionsCopy,numThread);

        numberOfClasses = this->UpdateClassesMemberships(fiberComputationData,previousDirections,m_Generators[numThread]);

        for (unsigned int i = 0;i < fiberComputationData.logParticleWeights.size();++i)
        {
            if (!std::isfinite(fiberComputationData.logParticleWeights[i]))
                itkExceptionMacro("Nan weights after update class membership");
        }

        // Continue only if some particles are still moving
        stopLoop = true;
        for (unsigned int i = 0;i < m_NumberOfParticles;++i)
        {
            if (fiberComputationData.fiberNumberOfPoints[i] == maxIterations)
                fiberComputationData.stoppedParticles[i] = true;

            if (!fiberComputationData.stoppedParticles[i])
            {
                stopLoop = false;
                break;
            }
        }
    }

    // Prepare backward loop
    stopLoop = true;
    FiberType invertedFiber;
    bool is2d = m_InputModelImage->GetLargestPossibleRegion().GetSize()[2] == 1;
    VectorType previousModelValue(m_ModelDimension);
    PointType previousPoint;
    ContinuousIndexType previousIndex;
    Vector3DType nullDirection(0.0), dummyDirection;

    for (unsigned int i = 0;i < m_NumberOfParticles;++i)
    {
        // Unstop particles
        if (fiberComputationData.fiberNumberOfPoints[i] <= maxIterations)
        {
            stopLoop = false;
            fiberComputationData.stoppedParticles[i] = false;
        }

        // Change fiber order, previous direction and log weight update
        invertedFiber.resize(fiberComputationData.fiberNumberOfPoints[i]);
        FiberType::reverse_iterator fiberItr;
        unsigned int pos = 0;
        for (fiberItr = fiberComputationData.fiberParticles[i].rbegin(); fiberItr != fiberComputationData.fiberParticles[i].rend(); ++fiberItr)
        {
            invertedFiber[pos] = *fiberItr;
            ++pos;
        }

        fiberComputationData.fiberParticles[i] = invertedFiber;

        unsigned int numPoints = fiberComputationData.fiberNumberOfPoints[i];
        unsigned int lastIndex = numPoints - 1;
        currentPoint = fiberComputationData.fiberParticles[i][lastIndex];
        m_SeedMask->TransformPhysicalPointToContinuousIndex(currentPoint,currentIndex);
        modelValue.Fill(0.0);
        previousPoint = fiberComputationData.fiberParticles[i][lastIndex];
        unsigned int oneToLastIndex = lastIndex;
        if (lastIndex > 0)
            oneToLastIndex = lastIndex - 1;

        previousPoint = fiberComputationData.fiberParticles[i][oneToLastIndex];
        m_SeedMask->TransformPhysicalPointToContinuousIndex(previousPoint,previousIndex);
        previousModelValue.Fill(0.0);

        this->ComputeModelValue(modelInterpolator, previousIndex, previousModelValue);
        this->ComputeModelValue(modelInterpolator, currentIndex, modelValue);

        if (fiberComputationData.fiberNumberOfPoints[i] <= 1)
        {
            previousDirections[i] = this->ProposeNewDirection(nullDirection, modelValue, m_Generators[numThread], numThread);

            if (previousDirections[i][1 + is2d] > 0)
                previousDirections[i] *= -1;

            fiberComputationData.previousUpdateLogWeights[i] = - std::log(m_NumberOfParticles);
        }
        else
        {
            previousDirections[i] = fiberComputationData.fiberParticles[i][lastIndex] - fiberComputationData.fiberParticles[i][oneToLastIndex];
            anima::Normalize(previousDirections[i],previousDirections[i]);

            dummyDirection = previousDirections[i];
            if (oneToLastIndex > 0)
                dummyDirection = fiberComputationData.fiberParticles[i][oneToLastIndex] - fiberComputationData.fiberParticles[i][oneToLastIndex - 1];
            anima::Normalize(dummyDirection,dummyDirection);

            estimatedB0Value = m_B0Interpolator->EvaluateAtContinuousIndex(currentIndex);
            estimatedNoiseValue = m_NoiseInterpolator->EvaluateAtContinuousIndex(currentIndex);

            fiberComputationData.previousUpdateLogWeights[i] = this->ComputeLogWeightUpdate(estimatedB0Value, estimatedNoiseValue, dummyDirection,
                                                                                            previousDirections[i], previousModelValue, modelValue, numThread);
        }
    }

    // Now that the fibers have been reversed and updated, perform the backward progression
    while (!stopLoop)
    {
        logWeightSums.resize(numberOfClasses);
        std::fill(logWeightSums.begin(),logWeightSums.end(),0.0);

        this->ProgressParticles(fiberComputationData,modelInterpolator,previousDirections,numThread);

        // Update weights
        this->UpdateWeightsFromCurrentData(fiberComputationData,logWeightSums);

        // Resampling if necessary
        this->CheckAndPerformOccasionalResampling(fiberComputationData,previousDirections,previousDirectionsCopy,numThread);

        numberOfClasses = this->UpdateClassesMemberships(fiberComputationData,previousDirections,m_Generators[numThread]);

        for (unsigned int i = 0;i < fiberComputationData.logParticleWeights.size();++i)
        {
            if (!std::isfinite(fiberComputationData.logParticleWeights[i]))
                itkExceptionMacro("Nan weights after update class membership");
        }

        // Continue only if some particles are still moving
        stopLoop = true;
        for (unsigned int i = 0;i < m_NumberOfParticles;++i)
        {
            if (fiberComputationData.fiberNumberOfPoints[i] == maxIterations)
                fiberComputationData.stoppedParticles[i] = true;

            if (!fiberComputationData.stoppedParticles[i])
            {
                stopLoop = false;
                break;
            }
        }
    }

    // Now that we're done, if we don't keep individual particles, merge them cluster by cluster
    if (m_MAPMergeFibers)
    {
        FiberProcessVectorType mergedOutput,classMergedOutput;
        for (unsigned int i = 0;i < numberOfClasses;++i)
        {
            this->MergeParticleClassFibers(fiberComputationData,classMergedOutput,i);
            mergedOutput.insert(mergedOutput.end(),classMergedOutput.begin(),classMergedOutput.end());
        }

        fiberComputationData.fiberParticles = mergedOutput;
    }

    if (m_MAPMergeFibers)
        resultWeights = fiberComputationData.logClassWeights;
    else
        resultWeights = fiberComputationData.logParticleWeights;

    return fiberComputationData.fiberParticles;
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::InitializeFirstIterationFromModel(VectorType &modelValue, unsigned int threadId,
                                    DirectionVectorType &initialDirections)
{
    initialDirections.resize(m_NumberOfParticles);
    Vector3DType nullDirection(0.0);
    bool is2d = (m_InputModelImage->GetLargestPossibleRegion().GetSize()[2] == 1);

    for (unsigned int i = 0;i < m_NumberOfParticles;++i)
    {
        initialDirections[i] = this->ProposeNewDirection(nullDirection, modelValue, m_Generators[threadId], threadId);

        if (initialDirections[i][1 + is2d] < 0)
            initialDirections[i] *= -1.0;

        if (is2d)
        {
            initialDirections[i][2] = 0.0;
            anima::Normalize(initialDirections[i],initialDirections[i]);
        }
    }
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::ProgressParticles(FiberWorkType &fiberComputationData, InterpolatorPointer &modelInterpolator,
                    DirectionVectorType &previousDirections, unsigned int numThread)
{
    double estimatedNoiseValue = 20.0;
    VectorType previousModelValue(m_ModelDimension);
    ContinuousIndexType currentIndex, newIndex;
    PointType currentPoint;
    IndexType closestIndex;
    VectorType modelValue(m_ModelDimension);
    Vector3DType newDirection;

    for (unsigned int i = 0;i < m_NumberOfParticles;++i)
    {
        // Do not compute trashed fibers
        if (fiberComputationData.stoppedParticles[i])
        {
            fiberComputationData.logParticleWeights[i] += fiberComputationData.previousUpdateLogWeights[i];
            continue;
        }

        currentPoint = fiberComputationData.fiberParticles[i].back();

        m_SeedMask->TransformPhysicalPointToContinuousIndex(currentPoint,currentIndex);

        // Trash fiber if it goes outside of the brain
        if (!modelInterpolator->IsInsideBuffer(currentIndex))
        {
            fiberComputationData.stoppedParticles[i] = true;
            fiberComputationData.logParticleWeights[i] += fiberComputationData.previousUpdateLogWeights[i];
            continue;
        }

        // Trash fiber if it goes through the cut mask
        m_SeedMask->TransformPhysicalPointToIndex(currentPoint,closestIndex);

        if (m_CutMask)
        {
            if (m_CutMask->GetPixel(closestIndex) != 0)
            {
                fiberComputationData.stoppedParticles[i] = true;
                fiberComputationData.logParticleWeights[i] += fiberComputationData.previousUpdateLogWeights[i];
                continue;
            }
        }

        // Computes diffusion information at current position
        modelValue.Fill(0.0);
        this->ComputeModelValue(modelInterpolator, currentIndex, modelValue);
        double estimatedB0Value = m_B0Interpolator->EvaluateAtContinuousIndex(currentIndex);
        estimatedNoiseValue = m_NoiseInterpolator->EvaluateAtContinuousIndex(currentIndex);

        if (!this->CheckModelProperties(estimatedB0Value,estimatedNoiseValue,modelValue,numThread))
        {
            fiberComputationData.stoppedParticles[i] = true;
            fiberComputationData.logParticleWeights[i] += fiberComputationData.previousUpdateLogWeights[i];
            continue;
        }

        // Propose a new direction based on the previous one and the diffusion information at current position
        newDirection = this->ProposeNewDirection(previousDirections[i], modelValue, m_Generators[numThread], numThread);

        // Update the position of the particle
        for (unsigned int j = 0;j < InputModelImageType::ImageDimension;++j)
            currentPoint[j] += m_StepProgression * newDirection[j];

        // Log-weight update must be done at new position (except for prior and proposal)
        m_SeedMask->TransformPhysicalPointToContinuousIndex(currentPoint,newIndex);

        // Set the new proposed direction as the current direction
        previousDirections[i] = newDirection;

        modelValue.Fill(0.0);

        if (!modelInterpolator->IsInsideBuffer(newIndex))
        {
            fiberComputationData.stoppedParticles[i] = true;
            fiberComputationData.logParticleWeights[i] += fiberComputationData.previousUpdateLogWeights[i];
            continue;
        }

        fiberComputationData.fiberParticles[i].push_back(currentPoint);
        fiberComputationData.fiberNumberOfPoints[i]++;

        previousModelValue = modelValue;
        this->ComputeModelValue(modelInterpolator, newIndex, modelValue);
        estimatedB0Value = m_B0Interpolator->EvaluateAtContinuousIndex(newIndex);
        estimatedNoiseValue = m_NoiseInterpolator->EvaluateAtContinuousIndex(newIndex);

        // Update the weight of the particle
        fiberComputationData.previousUpdateLogWeights[i] = this->ComputeLogWeightUpdate(estimatedB0Value, estimatedNoiseValue, previousDirections[i],
                                                                                        newDirection, previousModelValue, modelValue, numThread);

        fiberComputationData.logParticleWeights[i] += fiberComputationData.previousUpdateLogWeights[i];
    }
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::UpdateWeightsFromCurrentData(FiberWorkType &fiberComputationData, ListType &logWeightSums)
{
    ListType tmpVector;
    unsigned int numberOfClasses = logWeightSums.size();
    // Computes weight sum for further weight normalization
    for (unsigned int i = 0;i < numberOfClasses;++i)
    {
        unsigned int classSize = fiberComputationData.classSizes[i];
        tmpVector.resize(classSize);

        for (unsigned int j = 0;j < classSize;++j)
            tmpVector[j] = fiberComputationData.logParticleWeights[fiberComputationData.reverseClassMemberships[i][j]];

        logWeightSums[i] = anima::ExponentialSum(tmpVector);
    }

    // Weight normalization
    for (unsigned int i = 0;i < m_NumberOfParticles;++i)
        fiberComputationData.logNormalizedParticleWeights[i] = fiberComputationData.logParticleWeights[i] - logWeightSums[fiberComputationData.classMemberships[i]];

    // Class weights update
    tmpVector.resize(numberOfClasses);
    for (unsigned int i = 0;i < numberOfClasses;++i)
        tmpVector[i] = fiberComputationData.logClassWeights[i] + logWeightSums[i];

    double logSumClassWeights = anima::ExponentialSum(tmpVector);

    for (unsigned int i = 0;i < numberOfClasses;++i)
        fiberComputationData.logClassWeights[i] = fiberComputationData.logClassWeights[i] + logWeightSums[i] - logSumClassWeights;

    // For computational stability reasons, replace un-normalized weights by normalized ones at current step
    fiberComputationData.logParticleWeights = fiberComputationData.logNormalizedParticleWeights;
}

template <class TInputModelImageType>
void
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::CheckAndPerformOccasionalResampling(FiberWorkType &fiberComputationData, DirectionVectorType &previousDirections,
                                      DirectionVectorType &previousDirectionsCopy, unsigned int numThread)
{
    unsigned int numberOfClasses = fiberComputationData.classSizes.size();
    ListType weightSpecificClassValues, previousUpdateLogWeightsCopy;
    FiberProcessVectorType fiberParticlesCopy;
    std::vector <bool> fiberStoppedCopy;
    ListType tmpVector;
    std::vector <bool> usedFibers;

    // Resampling if necessary, done class by class
    for (unsigned int m = 0;m < numberOfClasses;++m)
    {
        unsigned int classSize = fiberComputationData.classSizes[m];
        tmpVector.resize(classSize);
        for (unsigned int i = 0;i < classSize;++i)
            tmpVector[i] = 2 * fiberComputationData.logNormalizedParticleWeights[fiberComputationData.reverseClassMemberships[m][i]];

        double logEffectiveNumberOfParticles = - anima::ExponentialSum(tmpVector);

        // Actual class resampling
        if (std::exp(logEffectiveNumberOfParticles) < m_ResamplingThreshold * classSize)
        {
            weightSpecificClassValues.resize(classSize);
            previousDirectionsCopy.resize(classSize);
            fiberParticlesCopy.resize(classSize);
            fiberStoppedCopy.resize(classSize);
            previousUpdateLogWeightsCopy.resize(classSize);

            for (unsigned int i = 0;i < classSize;++i)
            {
                unsigned int posIndex = fiberComputationData.reverseClassMemberships[m][i];
                weightSpecificClassValues[i] = std::exp(fiberComputationData.logNormalizedParticleWeights[posIndex]);
                previousDirectionsCopy[i] = previousDirections[posIndex];
                fiberParticlesCopy[i] = fiberComputationData.fiberParticles[posIndex];
                fiberStoppedCopy[i] = fiberComputationData.stoppedParticles[posIndex];
                previousUpdateLogWeightsCopy[i] = fiberComputationData.previousUpdateLogWeights[posIndex];
            }

            std::discrete_distribution<> dist(weightSpecificClassValues.begin(),weightSpecificClassValues.end());
            usedFibers.resize(classSize);
            std::fill(usedFibers.begin(),usedFibers.end(),false);

            for (unsigned int i = 0;i < classSize;++i)
            {
                unsigned int z = dist(m_Generators[numThread]);
                unsigned int iReal = fiberComputationData.reverseClassMemberships[m][i];
                previousDirections[iReal] = previousDirectionsCopy[z];
                fiberComputationData.fiberParticles[iReal] = fiberParticlesCopy[z];
                fiberComputationData.stoppedParticles[iReal] = fiberStoppedCopy[z];
                fiberComputationData.previousUpdateLogWeights[iReal] = previousUpdateLogWeightsCopy[z];
                usedFibers[z] = true;
            }

            // Update only weightVals, oldWeightVals will get updated when starting back the loop
            // Same here for stopped fibers, they get rejected when resampling
            for (unsigned int i = 0;i < classSize;++i)
            {
                fiberComputationData.logParticleWeights[fiberComputationData.reverseClassMemberships[m][i]] = std::log(- classSize);
                fiberComputationData.logNormalizedParticleWeights[fiberComputationData.reverseClassMemberships[m][i]] = std::log(- classSize);
            }
        }
    }
}

template <class TInputModelImageType>
unsigned int
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::UpdateClassesMemberships(FiberWorkType &fiberData, DirectionVectorType &directions, std::mt19937 &random_generator)
{
    const unsigned int p = PointType::PointDimension;
    typedef anima::KMeansFilter <PointType,p> KMeansFilterType;
    unsigned int numClasses = fiberData.classSizes.size();

    // Deciding on cluster merges
    FiberProcessVectorType mapMergedFibersRef, mapMergedFibersFlo;
    unsigned int newNumClasses = numClasses;

    MembershipType classesFusion(numClasses);
    for (unsigned int i = 0;i < numClasses;++i)
        classesFusion[i] = i;

    // As described in IPMI 2013, we take the input classes and first try to fuse them
    // This is based on a range of possible criterions specified by the user
    if (numClasses > 1)
    {
        for (unsigned int i = 0;i < numClasses;++i)
        {
            // Fuse test is done on average cluster fiber, if it is an active cluster,
            // i.e. at least one of its particle is still moving
            bool activeClass = this->MergeParticleClassFibers(fiberData,mapMergedFibersRef,i);
            if (!activeClass)
                continue;

            for (unsigned int j = i+1;j < numClasses;++j)
            {
                if (classesFusion[j] != j)
                    continue;

                double maxVal = 0;
                bool activeSubClass = this->MergeParticleClassFibers(fiberData,mapMergedFibersFlo,j);
                if (!activeSubClass)
                    continue;

                // Compute a distance between the two clusters, based on user input
                switch (m_ClusterDistance)
                {
                    case 0:
                    {
                        // Hausdorff distance
                        for (unsigned int l = 0;l < mapMergedFibersRef[0].size();++l)
                        {
                            double tmpVal = anima::ComputePointToSetDistance(mapMergedFibersRef[0][l], mapMergedFibersFlo[0]);

                            if (tmpVal > maxVal)
                                maxVal = tmpVal;

                            if (maxVal > m_PositionDistanceFuseThreshold)
                                break;
                        }

                        if (maxVal <= m_PositionDistanceFuseThreshold)
                        {
                            for (unsigned int l = 0;l < mapMergedFibersFlo[0].size();++l)
                            {
                                double tmpVal = anima::ComputePointToSetDistance(mapMergedFibersFlo[0][l], mapMergedFibersRef[0]);

                                if (tmpVal > maxVal)
                                    maxVal = tmpVal;

                                if (maxVal > m_PositionDistanceFuseThreshold)
                                    break;
                            }
                        }

                        break;
                    }

                    case 1:
                    {
                        // Modified Hausdorff distance
                        maxVal = anima::ComputeModifiedDirectedHausdorffDistance(mapMergedFibersRef[0], mapMergedFibersFlo[0]);

                        if (maxVal <= m_PositionDistanceFuseThreshold)
                            maxVal = std::max(maxVal, anima::ComputeModifiedDirectedHausdorffDistance(mapMergedFibersFlo[0], mapMergedFibersRef[0]));

                        break;
                    }

                    default:
                        break;
                }

                // If computed distance is smaller than a threshold, we fuse
                // To do so, an index table (classesFusion) is updated, each of its cells tells
                // to which new class the current class belongs. newNumClasses is the new number of classes
                if (maxVal <= m_PositionDistanceFuseThreshold)
                {
                    newNumClasses--;
                    classesFusion[j] = classesFusion[i];
                }
            }
        }

        // Some post-processing to have contiguous class numbers as an output
        // mapFusion will hold the correspondance between non contiguous and contiguous indexes
        int maxVal = -1;
        unsigned int currentIndex = 0;
        std::map <unsigned int, unsigned int> mapFusion;
        for (unsigned int i = 0;i < numClasses;++i)
        {
            if (maxVal < (int)classesFusion[i])
            {
                mapFusion.insert(std::make_pair(classesFusion[i],currentIndex));
                ++currentIndex;
                maxVal = classesFusion[i];
            }
        }

        for (unsigned int i = 0;i < numClasses;++i)
            classesFusion[i] = mapFusion[classesFusion[i]];
    }

    std::vector <MembershipType> fusedClassesIndexes(newNumClasses);
    for (unsigned int i = 0;i < numClasses;++i)
        fusedClassesIndexes[classesFusion[i]].push_back(i);

    // Now we're done with selecting what to fuse
    // Therefore, deciding on cluster splits. The trick here is that we don't want to actually really perform fuse
    // if it is to split right after (for speed reasons). So we'll play with classesFusion indexes all along
    // to keep track of the original indexes.
    DirectionVectorType afterMergeClassesDirections(newNumClasses);
    MembershipType afterMergeNumPoints(newNumClasses,0);
    std::vector <bool> splitClasses(newNumClasses,false);

    Vector3DType zeroDirection(0.0);
    std::fill(afterMergeClassesDirections.begin(),afterMergeClassesDirections.end(),zeroDirection);

    // Compute average directions after fusion, directions contains the last directions taken by particles
    for (unsigned int i = 0;i < m_NumberOfParticles;++i)
    {
        if (std::exp(fiberData.logNormalizedParticleWeights[i]) == 0)
            continue;

        unsigned int classIndex = classesFusion[fiberData.classMemberships[i]];
        for (unsigned int j = 0;j < p;++j)
            afterMergeClassesDirections[classIndex][j] += directions[i][j];
        afterMergeNumPoints[classIndex]++;
    }

    unsigned int numSplits = 0;
    std::vector < std::pair <unsigned int, double> > afterMergeKappaValues;
    // From those averaged directions, we compute a dispersion kappa value, that will be used to decide on split
    for (unsigned int i = 0;i < newNumClasses;++i)
    {
        double norm = 0;
        for (unsigned int j = 0;j < p;++j)
            norm += afterMergeClassesDirections[i][j] * afterMergeClassesDirections[i][j];
        norm = sqrt(norm);

        double R = 1.0;
        double kappa = m_KappaSplitThreshold + 1;
        if (afterMergeNumPoints[i] != 0)
        {
            R = norm / afterMergeNumPoints[i];

            if (R*R > 1.0 - 1.0e-16)
                R = sqrt(1.0 - 1.0e-16);

            kappa = std::exp( anima::safe_log(R) + anima::safe_log(p-R*R) - anima::safe_log(1-R*R) );
        }

        afterMergeKappaValues.push_back(std::make_pair(i,kappa));
        // We do not allow any split resulting in less than m_MinimalNumberOfParticlesPerClass
        // so testing with respect to 2*m_MinimalNumberOfParticlesPerClass
        // If it is ok, and kappa is small enough, there is too much dispersion inside the cluster -> splitting
        if ((kappa <= m_KappaSplitThreshold)&&(afterMergeNumPoints[i] >= 2 * m_MinimalNumberOfParticlesPerClass)&&(afterMergeNumPoints[i] != 0))
            numSplits++;
        else
            afterMergeKappaValues[i].second = m_KappaSplitThreshold + 1;
    }

    std::partial_sort(afterMergeKappaValues.begin(),afterMergeKappaValues.begin() + numSplits,afterMergeKappaValues.end(),pair_comparator());

    for (unsigned int i = 0;i < numSplits;++i)
        splitClasses[afterMergeKappaValues[i].first] = true;

    // Finally apply all this to get our final clusters
    // Each split class will be split into two, so new number of classes
    // after merge and split is newNumClasses + numSplits
    unsigned int finalNumClasses = newNumClasses + numSplits;

    MembershipType newClassesMemberships(m_NumberOfParticles,0);
    std::vector <MembershipType> newReverseClassesMemberships(finalNumClasses);
    MembershipType newClassSizes(finalNumClasses,0);
    ListType newLogNormalizedParticleWeights = fiberData.logNormalizedParticleWeights;
    ListType newLogClassWeights(finalNumClasses,0);
    ListType logSumVector, logSumVectorNewClass;

    unsigned int currentIndex = 0;

    FiberType vectorToCluster;
    MembershipType clustering;

    // Now, do the real merge/split part
    for (unsigned int i = 0;i < newNumClasses;++i)
    {
        if (!splitClasses[i])
        {
            // ith class is just a potential merge of classes. Easy case: just take all particles
            // from classes marked as new ith class
            for (unsigned int j = 0;j < fusedClassesIndexes[i].size();++j)
            {
                unsigned int classIndex = fusedClassesIndexes[i][j];
                for (unsigned int k = 0;k < fiberData.reverseClassMemberships[classIndex].size();++k)
                {
                    unsigned int particleNumber = fiberData.reverseClassMemberships[classIndex][k];
                    newClassesMemberships[particleNumber] = currentIndex;
                    newReverseClassesMemberships[currentIndex].push_back(particleNumber);
                }
            }

            if (fusedClassesIndexes[i].size() != 1)
            {
                // Recompute class weights after fusion
                logSumVector.clear();
                for (unsigned int j = 0;j < fusedClassesIndexes[i].size();++j)
                {
                    unsigned int classIndex = fusedClassesIndexes[i][j];
                    for (unsigned int k = 0;k < fiberData.reverseClassMemberships[classIndex].size();++k)
                        logSumVector.push_back(fiberData.logClassWeights[classIndex] * fiberData.logNormalizedParticleWeights[fiberData.reverseClassMemberships[classIndex][k]]);
                }

                newLogClassWeights[currentIndex] = anima::ExponentialSum(logSumVector);

                // Recompute particle weights after fusion
                for (unsigned int j = 0;j < fusedClassesIndexes[i].size();++j)
                {
                    unsigned int classIndex = fusedClassesIndexes[i][j];
                    for (unsigned int k = 0;k < fiberData.reverseClassMemberships[classIndex].size();++k)
                    {
                        unsigned int posIndex = fiberData.reverseClassMemberships[classIndex][k];
                        newLogNormalizedParticleWeights[posIndex] = fiberData.logClassWeights[classIndex] + fiberData.logNormalizedParticleWeights[posIndex] - newLogClassWeights[currentIndex];
                    }
                }
            }
            else
                newLogClassWeights[currentIndex] = fiberData.logClassWeights[fusedClassesIndexes[i][0]];

            ++currentIndex;
        }
        else
        {
            // ith class is a split at the end. In that case, first gather all particles
            // from classes merged before. Then, plug k-means
            vectorToCluster.clear();

            // Gather particles
            for (unsigned int j = 0;j < fusedClassesIndexes[i].size();++j)
            {
                unsigned int classIndex = fusedClassesIndexes[i][j];
                for (unsigned int k = 0;k < fiberData.reverseClassMemberships[classIndex].size();++k)
                    vectorToCluster.push_back(fiberData.fiberParticles[fiberData.reverseClassMemberships[classIndex][k]].back());
            }

            clustering.resize(vectorToCluster.size());

            // Now cluster them, loop until the two classes are not empty
            bool loopOnClustering = true;
            std::uniform_int_distribution <unsigned int> uniInt(0,1);
            while (loopOnClustering)
            {
                for (unsigned int j = 0;j < clustering.size();++j)
                    clustering[j] = uniInt(random_generator) % 2;

                KMeansFilterType kmFilter;
                kmFilter.SetInputData(vectorToCluster);
                kmFilter.SetNumberOfClasses(2);
                kmFilter.InitializeClassesMemberships(clustering);
                kmFilter.SetMaxIterations(100);
                kmFilter.SetVerbose(false);

                kmFilter.Update();

                // Otherwise, go ahead and do the splitting
                clustering = kmFilter.GetClassesMemberships();

                if ((kmFilter.GetNumberPerClass(0) > 0)&&(kmFilter.GetNumberPerClass(1) > 0))
                    loopOnClustering = false;
            }

            // Now assigne new class indexes to particles, plus update class weights
            unsigned int newClassIndex = currentIndex + 1;

            logSumVector.clear();
            logSumVectorNewClass.clear();

            unsigned int pos = 0;
            for (unsigned int j = 0;j < fusedClassesIndexes[i].size();++j)
            {
                unsigned int classIndex = fusedClassesIndexes[i][j];
                for (unsigned int k = 0;k < fiberData.reverseClassMemberships[classIndex].size();++k)
                {
                    unsigned int classPos = currentIndex + clustering[pos];

                    newClassesMemberships[fiberData.reverseClassMemberships[classIndex][k]] = classPos;
                    newReverseClassesMemberships[classPos].push_back(fiberData.reverseClassMemberships[classIndex][k]);

                    if (clustering[pos] == 0)
                        logSumVector.push_back(fiberData.logClassWeights[classIndex] + fiberData.logNormalizedParticleWeights[fiberData.reverseClassMemberships[classIndex][k]]);
                    else
                        logSumVectorNewClass.push_back(fiberData.logClassWeights[classIndex] + fiberData.logNormalizedParticleWeights[fiberData.reverseClassMemberships[classIndex][k]]);

                    ++pos;
                }
            }

            newLogClassWeights[currentIndex] = anima::ExponentialSum(logSumVector);
            newLogClassWeights[newClassIndex] = anima::ExponentialSum(logSumVectorNewClass);

            // Finally, update particle weights
            pos = 0;
            for (unsigned int j = 0;j < fusedClassesIndexes[i].size();++j)
            {
                unsigned int classIndex = fusedClassesIndexes[i][j];
                for (unsigned int k = 0;k < fiberData.reverseClassMemberships[classIndex].size();++k)
                {
                    unsigned int classPos = currentIndex + clustering[pos];

                    unsigned int posIndex = fiberData.reverseClassMemberships[classIndex][k];
                    newLogNormalizedParticleWeights[posIndex] = fiberData.logClassWeights[classIndex] + fiberData.logNormalizedParticleWeights[posIndex] - newLogClassWeights[classPos];

                    ++pos;
                }
            }

            currentIndex += 2;
        }
    }

    for (unsigned int i = 0;i < finalNumClasses;++i)
        newClassSizes[i] = newReverseClassesMemberships[i].size();

    // Replace all fiber data by new ones computed here and we're done
    fiberData.classSizes = newClassSizes;
    fiberData.logClassWeights = newLogClassWeights;
    fiberData.classMemberships = newClassesMemberships;
    fiberData.logNormalizedParticleWeights = newLogNormalizedParticleWeights;
    fiberData.reverseClassMemberships = newReverseClassesMemberships;

    return finalNumClasses;
}

template <class TInputModelImageType>
bool
BaseProbabilisticTractographyImageFilter <TInputModelImageType>
::MergeParticleClassFibers(FiberWorkType &fiberData, FiberProcessVectorType &outputMerged, unsigned int classNumber)
{
    unsigned int numClasses = fiberData.classSizes.size();
    outputMerged.clear();
    if (classNumber >= numClasses)
        return false;

    outputMerged.resize(1);

    std::vector <unsigned int> runningIndexes, stoppedIndexes;
    for (unsigned int j = 0;j < fiberData.classSizes[classNumber];++j)
    {
        if (fiberData.stoppedParticles[fiberData.reverseClassMemberships[classNumber][j]])
            stoppedIndexes.push_back(fiberData.reverseClassMemberships[classNumber][j]);
        else
            runningIndexes.push_back(fiberData.reverseClassMemberships[classNumber][j]);
    }

    FiberType classFiber;
    FiberType tmpFiber;
    ListType sumWeights;
    unsigned int sizeMerged = 0;
    unsigned int p = PointType::GetPointDimension();

    if (runningIndexes.size() != 0)
    {
        // Use weights provided
        for (unsigned int j = 0;j < runningIndexes.size();++j)
        {
            double tmpWeight = std::exp(fiberData.logParticleWeights[runningIndexes[j]]);
            if (tmpWeight <= 0)
                continue;

            tmpFiber = fiberData.fiberParticles[runningIndexes[j]];
            for (unsigned int k = 0;k < tmpFiber.size();++k)
            {
                if (k < sizeMerged)
                {
                    for (unsigned int l = 0;l < p;++l)
                        classFiber[k][l] += tmpWeight * tmpFiber[k][l];
                    sumWeights[k] += tmpWeight;
                }
                else
                {
                    sizeMerged++;
                    classFiber.push_back(tmpFiber[k]);
                    sumWeights.push_back(tmpWeight);
                    for (unsigned int l = 0;l < p;++l)
                        classFiber[k][l] *= tmpWeight;
                }
            }
        }

        for (unsigned int j = 0;j < sizeMerged;++j)
        {
            for (unsigned int k = 0;k < p;++k)
                classFiber[j][k] /= sumWeights[j];
        }

        outputMerged[0] = classFiber;
        return true;
    }

    // Treat all fibers equivalently, first construct groups of equal lengths
    std::vector < std::vector <unsigned int> > particleGroups;
    std::vector <unsigned int> particleSizes;
    for (unsigned int i = 0;i < stoppedIndexes.size();++i)
    {
        unsigned int particleSize = fiberData.fiberParticles[stoppedIndexes[i]].size();
        bool sizeFound = false;
        for (unsigned int j = 0;j < particleSizes.size();++j)
        {
            if (particleSize == particleSizes[j])
            {
                particleGroups[j].push_back(stoppedIndexes[i]);
                sizeFound = true;
                break;
            }
        }

        if (!sizeFound)
        {
            particleSizes.push_back(particleSize);
            std::vector <unsigned int> tmpVec(1,stoppedIndexes[i]);
            particleGroups.push_back(tmpVec);
        }
    }

    // For each group of equal length, build fiber
    outputMerged.resize(particleGroups.size());
    for (unsigned int i = 0;i < particleGroups.size();++i)
    {
        classFiber.clear();
        sizeMerged = 0;

        for (unsigned int j = 0;j < particleGroups[i].size();++j)
        {
            tmpFiber = fiberData.fiberParticles[particleGroups[i][j]];
            for (unsigned int k = 0;k < tmpFiber.size();++k)
            {
                if (k < sizeMerged)
                {
                    for (unsigned int l = 0;l < p;++l)
                        classFiber[k][l] += tmpFiber[k][l];
                }
                else
                {
                    sizeMerged++;
                    classFiber.push_back(tmpFiber[k]);
                }
            }
        }

        for (unsigned int j = 0;j < sizeMerged;++j)
        {
            for (unsigned int k = 0;k < p;++k)
                classFiber[j][k] /= particleGroups[i].size();
        }

        outputMerged[i] = classFiber;
    }

    return false;
}

} // end of namespace anima
