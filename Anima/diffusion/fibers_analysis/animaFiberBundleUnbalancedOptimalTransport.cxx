#include <animaFiberBundleUnbalancedOptimalTransport.h>

#include <animaVectorOperations.h>

#include <itkTimeProbe.h>
#include <itkPoolMultiThreader.h>

#include <vtkGenericCell.h>

namespace anima
{

FiberBundleUnbalancedOptimalTransport::FiberBundleUnbalancedOptimalTransport()
{
    m_MemorySizeLimit = 8.0;
    m_RhoValue = 1.0;
    m_EpsilonValue = std::sqrt(0.07);
    m_RelativeStopCriterion = 1.0e-4;

    m_AlphaValue = 1.0;
    m_KValue = 4.0;

    m_Verbose = true;
}

void
FiberBundleUnbalancedOptimalTransport
::Update()
{
    m_WassersteinSquaredDistance = 0.0;
    this->GetMultiThreader()->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());

    this->PrepareInputFibersData();

    m_WorkVector.resize(this->GetNumberOfWorkUnits());
    m_DistanceComputers.resize(this->GetNumberOfWorkUnits());
    for (unsigned int i = 0;i < this->GetNumberOfWorkUnits();++i)
    {
        m_DistanceComputers[i] = FiberDistanceComputerType::New();
        m_DistanceComputers[i]->SetFirstDataset(m_FirstDataset);
        m_DistanceComputers[i]->SetSecondDataset(m_SecondDataset);
        m_DistanceComputers[i]->SetSegmentLengthsFirstDataset(m_SegmentLengthsFirstDataset);
        m_DistanceComputers[i]->SetSegmentLengthsSecondDataset(m_SegmentLengthsSecondDataset);
        m_DistanceComputers[i]->SetSegmentTangentsFirstDataset(m_SegmentTangentsFirstDataset);
        m_DistanceComputers[i]->SetSegmentTangentsSecondDataset(m_SegmentTangentsSecondDataset);
        m_DistanceComputers[i]->SetKValue(m_KValue);
        m_DistanceComputers[i]->SetAlphaValue(m_AlphaValue);
        m_DistanceComputers[i]->SetEpsilonValue(m_EpsilonValue);
        m_DistanceComputers[i]->SetRhoValue(m_RhoValue);
        m_DistanceComputers[i]->SetNumberOfWorkUnits(1);
        m_DistanceComputers[i]->SetMemorySizeLimit(0.25);
        m_DistanceComputers[i]->SetVerbose(false);
    }

    unsigned int nbCellsFirst = m_FirstDataset->GetNumberOfCells();
    unsigned int nbCellsSecond = m_SecondDataset->GetNumberOfCells();

    if (m_Verbose)
        std::cout << "Number of fibers: ref: " << nbCellsFirst << ", moving: " << nbCellsSecond << std::endl;

    double dataSize = nbCellsFirst * nbCellsSecond * sizeof(double) / std::pow(1024.0,3);

    if (m_Verbose)
        std::cout << "Required memory to precompute distance matrix: " << dataSize << " Gb" << std::endl;

    bool precomputeDistanceMatrix = (dataSize < m_MemorySizeLimit);

    if (m_Verbose)
        std::cout << "Precomputing? " << (precomputeDistanceMatrix ? "Yes" : "No") << std::endl;

    if (precomputeDistanceMatrix)
    {
        this->PrecomputeDistanceMatrix();

        if (m_Verbose)
            std::cout << "Distance matrix precomputed" << std::endl;
    }
    else
        m_DistanceMatrix.set_size(0,0);

    // Now go on with the Sinkhorn algorithm
    m_UVector.resize(nbCellsFirst);
    m_VVector.resize(nbCellsSecond);

    std::fill(m_UVector.begin(), m_UVector.end(), 0.0);
    std::fill(m_VVector.begin(), m_VVector.end(), 0.0);

    double maxDiff = m_RelativeStopCriterion + 1.0;

    itk::TimeProbe tmpTime;
    tmpTime.Start();

    unsigned int numIterations = 0;
    while (maxDiff > m_RelativeStopCriterion)
    {
        ++numIterations;

        m_OldUVector = m_UVector;
        m_OldVVector = m_VVector;

        ThreadArguments tmpStr;
        tmpStr.uotPtr = this;

        // Update U vector
        m_UpdateUVector = true;
        this->GetMultiThreader()->SetSingleMethod(ThreadVectorUpdate,&tmpStr);
        this->GetMultiThreader()->SingleMethodExecute();

        // Update V vector
        m_UpdateUVector = false;
        this->GetMultiThreader()->SetSingleMethod(ThreadVectorUpdate,&tmpStr);
        this->GetMultiThreader()->SingleMethodExecute();

        maxDiff = 0.0;
        for (unsigned int i = 0;i < nbCellsFirst;++i)
        {
            double diffVal = std::abs(m_OldUVector[i] - m_UVector[i]) / std::max(std::abs(m_OldUVector[i]),std::abs(m_UVector[i]));
            if (diffVal > maxDiff)
                maxDiff = diffVal;
        }

        for (unsigned int i = 0;i < nbCellsSecond;++i)
        {
            double diffVal = std::abs(m_OldVVector[i] - m_VVector[i]) / std::max(std::abs(m_OldVVector[i]),std::abs(m_VVector[i]));
            if (diffVal > maxDiff)
                maxDiff = diffVal;
        }
    }

    tmpTime.Stop();

    if (m_Verbose)
        std::cout << "Number of iterations: " << numIterations << ". Computation time: " << tmpTime.GetTotal() << "s..." << std::endl;

    this->ComputeWassersteinDistanceFomData();
}

void
FiberBundleUnbalancedOptimalTransport
::PrepareInputFibersData()
{
    unsigned int nbTotalPtsFirst = m_FirstDataset->GetNumberOfPoints();
    unsigned int nbTotalPtsSecond = m_SecondDataset->GetNumberOfPoints();

    m_SegmentLengthsFirstDataset = vtkDoubleArray::New();
    m_SegmentLengthsFirstDataset->SetNumberOfComponents(1);
    m_SegmentLengthsFirstDataset->SetNumberOfTuples(nbTotalPtsFirst);
    m_SegmentLengthsFirstDataset->SetName("Lengths");

    m_SegmentLengthsSecondDataset = vtkDoubleArray::New();
    m_SegmentLengthsSecondDataset->SetNumberOfComponents(1);
    m_SegmentLengthsSecondDataset->SetNumberOfTuples(nbTotalPtsSecond);
    m_SegmentLengthsSecondDataset->SetName("Lengths");

    m_SegmentTangentsFirstDataset = vtkDoubleArray::New();
    m_SegmentTangentsFirstDataset->SetNumberOfComponents(3);
    m_SegmentTangentsFirstDataset->SetNumberOfTuples(nbTotalPtsFirst);
    m_SegmentTangentsFirstDataset->SetName("Tangents");

    m_SegmentTangentsSecondDataset = vtkDoubleArray::New();
    m_SegmentTangentsSecondDataset->SetNumberOfComponents(3);
    m_SegmentTangentsSecondDataset->SetNumberOfTuples(nbTotalPtsSecond);
    m_SegmentTangentsSecondDataset->SetName("Tangents");

    ThreadArguments tmpStr;
    tmpStr.uotPtr = this;

    this->GetMultiThreader()->SetSingleMethod(ThreadPrepare, &tmpStr);
    this->GetMultiThreader()->SingleMethodExecute();
}

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
FiberBundleUnbalancedOptimalTransport
::ThreadPrepare(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;
    unsigned int numTotalThread = threadArgs->NumberOfWorkUnits;

    ThreadArguments *tmpArg = (ThreadArguments *)threadArgs->UserData;
    unsigned int nbTotalCellsFirst = tmpArg->uotPtr->GetFirstDataset()->GetNumberOfCells();
    unsigned int nbTotalCellsSecond = tmpArg->uotPtr->GetSecondDataset()->GetNumberOfCells();

    unsigned int step = nbTotalCellsFirst / numTotalThread;
    unsigned int startIndex = nbThread * step;
    unsigned int endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbTotalCellsFirst;

    vtkGenericCell *cellData = vtkGenericCell::New();

    for (unsigned int i = startIndex;i < endIndex;++i)
        tmpArg->uotPtr->ComputeExtraDataOnCell(i,cellData,0);

    step = nbTotalCellsSecond / numTotalThread;
    startIndex = nbThread * step;
    endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbTotalCellsSecond;

    for (unsigned int i = startIndex;i < endIndex;++i)
        tmpArg->uotPtr->ComputeExtraDataOnCell(i,cellData,1);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

void
FiberBundleUnbalancedOptimalTransport
::ComputeExtraDataOnCell(unsigned int cellIndex, vtkGenericCell *cellData, unsigned int dataIndex)
{
    if (dataIndex == 0)
        m_FirstDataset->GetCell(cellIndex, cellData);
    else
        m_SecondDataset->GetCell(cellIndex, cellData);

    vtkPoints *cellPts = cellData->GetPoints();
    vtkIdType nbOfCellPts = cellPts->GetNumberOfPoints();

    double positionBefore[3];
    double positionAfter[3];
    double tangent[3];

    if (nbOfCellPts == 0)
        return;

    double length = 0;
    for (vtkIdType j = 0;j < nbOfCellPts;++j)
    {
        double norm = 0;

        vtkIdType indexBefore = j - 1;
        if (j - 1 < 0)
            indexBefore = j;
        vtkIdType indexAfter = std::min(nbOfCellPts - 1, j + 1);
        if (indexAfter == j)
            indexBefore = j - 1;

        int ptId = cellData->GetPointId(j);

        cellPts->GetPoint(indexBefore, positionBefore);
        cellPts->GetPoint(indexAfter, positionAfter);

        for (unsigned int k = 0;k < 3;++k)
        {
            tangent[k] = positionAfter[k] - positionBefore[k];
            norm += tangent[k] * tangent[k];
        }

        norm = std::sqrt(norm);

        if (indexAfter != j)
        {
            cellPts->GetPoint(j, positionBefore);
            length = 0;
            for (unsigned int k = 0;k < 3;++k)
                length += (positionAfter[k] - positionBefore[k]) * (positionAfter[k] - positionBefore[k]);

            length = std::sqrt(length);
        }

        if (norm != 0.0)
        {
            for (unsigned int k = 0;k < 3;++k)
                tangent[k] /= norm;
        }

        if (dataIndex == 0)
        {
            m_SegmentTangentsFirstDataset->SetTuple3(ptId, tangent[0], tangent[1], tangent[2]);
            m_SegmentLengthsFirstDataset->SetValue(ptId,length);
        }
        else
        {
            m_SegmentTangentsSecondDataset->SetTuple3(ptId, tangent[0], tangent[1], tangent[2]);
            m_SegmentLengthsSecondDataset->SetValue(ptId,length);
        }
    }
}

void
FiberBundleUnbalancedOptimalTransport
::PrecomputeDistanceMatrix()
{
    unsigned int nbCellsFirst = m_FirstDataset->GetNumberOfCells();
    unsigned int nbCellsSecond = m_SecondDataset->GetNumberOfCells();

    m_DistanceMatrix.set_size(nbCellsFirst, nbCellsSecond);

    ThreadArguments tmpStr;
    tmpStr.uotPtr = this;

    this->GetMultiThreader()->SetSingleMethod(ThreadPrecomputeDistanceMatrix, &tmpStr);
    this->GetMultiThreader()->SingleMethodExecute();
}

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
FiberBundleUnbalancedOptimalTransport
::ThreadPrecomputeDistanceMatrix(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;
    unsigned int numTotalThread = threadArgs->NumberOfWorkUnits;

    ThreadArguments *tmpArg = (ThreadArguments *)threadArgs->UserData;
    unsigned int nbTotalCellsFirst = tmpArg->uotPtr->GetFirstDataset()->GetNumberOfCells();

    unsigned int step = nbTotalCellsFirst / numTotalThread;
    unsigned int startIndex = nbThread * step;
    unsigned int endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbTotalCellsFirst;

    tmpArg->uotPtr->PrecomputeDistancesOnRange(startIndex, endIndex, nbThread);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

void
FiberBundleUnbalancedOptimalTransport
::PrecomputeDistancesOnRange(unsigned int startIndex, unsigned int endIndex, unsigned int threadId)
{
    unsigned int nbCellsSecond = m_SecondDataset->GetNumberOfCells();

    for (unsigned int i = startIndex;i < endIndex;++i)
    {
        for (unsigned int j = 0;j < nbCellsSecond;++j)
            m_DistanceMatrix(i,j) = this->ComputeDistance(i,j, threadId);
    }
}

double
FiberBundleUnbalancedOptimalTransport
::ComputeDistance(unsigned int firstIndex, unsigned int secondIndex, unsigned int threadId)
{
    m_DistanceComputers[threadId]->SetFiberIndexInFirstDataset(firstIndex);
    m_DistanceComputers[threadId]->SetFiberIndexInSecondDataset(secondIndex);

    m_DistanceComputers[threadId]->Update();

    return m_DistanceComputers[threadId]->GetWassersteinSquaredDistance();
}

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
FiberBundleUnbalancedOptimalTransport
::ThreadVectorUpdate(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;
    unsigned int numTotalThread = threadArgs->NumberOfWorkUnits;

    ThreadArguments *tmpArg = (ThreadArguments *)threadArgs->UserData;
    unsigned int nbCells = 0;

    if (tmpArg->uotPtr->GetUpdateUVector())
        nbCells = tmpArg->uotPtr->GetFirstDataset()->GetNumberOfCells();
    else
        nbCells = tmpArg->uotPtr->GetSecondDataset()->GetNumberOfCells();

    unsigned int step = nbCells / numTotalThread;
    unsigned int startIndex = nbThread * step;
    unsigned int endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbCells;

    tmpArg->uotPtr->ComputeVectorUpdateOnRange(startIndex, endIndex, nbThread);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

void
FiberBundleUnbalancedOptimalTransport
::ComputeVectorUpdateOnRange(unsigned int startIndex, unsigned int endIndex, unsigned int threadId)
{
    unsigned int nbCells = 0;
    unsigned int nbOtherCells = 0;

    std::vector <double> *updatedVector = &m_UVector;
    std::vector <double> *fixedVector = &m_VVector;

    if (m_UpdateUVector)
    {
        nbCells = m_FirstDataset->GetNumberOfCells();
        nbOtherCells = m_SecondDataset->GetNumberOfCells();
    }
    else
    {
        nbCells = m_SecondDataset->GetNumberOfCells();
        nbOtherCells = m_FirstDataset->GetNumberOfCells();

        updatedVector = &m_VVector;
        fixedVector = &m_UVector;
    }

    m_WorkVector[threadId].resize(nbOtherCells);

    bool useDistMatrix = (m_DistanceMatrix.rows() != 0);
    bool transposeDistMatrix = (!m_UpdateUVector);
    double lambda = m_RhoValue / (m_RhoValue + m_EpsilonValue);

    for (unsigned int i = startIndex;i < endIndex;++i)
    {
        double weightValue = 1.0;
        updatedVector->operator[](i) = lambda * std::log(weightValue);

        for (unsigned int j = 0;j < nbOtherCells;++j)
        {
            double distance = 0.0;
            if (useDistMatrix)
            {
                if (!transposeDistMatrix)
                    distance = m_DistanceMatrix.get(i,j);
                else
                    distance = m_DistanceMatrix.get(j,i);
            }
            else
            {
                if (m_UpdateUVector)
                    distance = this->ComputeDistance(i,j,threadId);
                else
                    distance = this->ComputeDistance(j,i,threadId);
            }

            m_WorkVector[threadId][j] = fixedVector->operator[](j) - distance / m_EpsilonValue;
        }

        updatedVector->operator[](i) -= lambda * anima::ExponentialSum(m_WorkVector[threadId]);
    }
}

void
FiberBundleUnbalancedOptimalTransport
::ComputeWassersteinDistanceFomData()
{
    unsigned int nbCellsFirst = m_FirstDataset->GetNumberOfCells();
    unsigned int nbCellsSecond = m_SecondDataset->GetNumberOfCells();

    m_WassersteinSquaredDistance = 0.0;
    std::vector <double> distKLUVector(nbCellsFirst, 0.0);
    std::vector <double> distKLVVector(nbCellsSecond, 0.0);

    bool useDistanceMatrix = (m_DistanceMatrix.rows() != 0);

    for (unsigned int i = 0;i < nbCellsFirst;++i)
    {
        for (unsigned int j = 0;j < nbCellsSecond;++j)
        {
            double ijDistance = 0.0;
            if (useDistanceMatrix)
                ijDistance = m_DistanceMatrix(i,j);
            else
                ijDistance = this->ComputeDistance(i,j,0);

            double Pij = std::exp(m_UVector[i] + m_VVector[j] - ijDistance / m_EpsilonValue);
            m_WassersteinSquaredDistance += Pij * ijDistance;
            distKLUVector[i] += Pij;
            distKLVVector[j] += Pij;
        }
    }

    for (unsigned int i = 0;i < nbCellsFirst;++i)
    {
        double aValue = 1.0;
        double xlnx = 0.0;
        if (distKLUVector[i] > 0)
            xlnx = distKLUVector[i] * std::log(distKLUVector[i] / aValue);

        m_WassersteinSquaredDistance += m_RhoValue * (xlnx - distKLUVector[i] + aValue);
    }

    for (unsigned int i = 0;i < nbCellsSecond;++i)
    {
        double bValue = 1.0;
        double xlnx = 0.0;
        if (distKLVVector[i] > 0)
            xlnx = distKLVVector[i] * std::log(distKLVVector[i] / bValue);

        m_WassersteinSquaredDistance += m_RhoValue * (xlnx - distKLVVector[i] + bValue);
    }
}

} // end namespace anima
