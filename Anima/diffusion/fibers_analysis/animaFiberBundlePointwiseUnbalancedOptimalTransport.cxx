#include <animaFiberBundlePointwiseUnbalancedOptimalTransport.h>

#include <animaVectorOperations.h>

#include <itkTimeProbe.h>
#include <itkPoolMultiThreader.h>

#include <vtkGenericCell.h>

namespace anima
{

FiberBundlePointwiseUnbalancedOptimalTransport
::FiberBundlePointwiseUnbalancedOptimalTransport()
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
FiberBundlePointwiseUnbalancedOptimalTransport
::Update()
{
    this->GetMultiThreader()->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
    m_WassersteinSquaredDistance = 0.0;

    this->PrepareInputFibersData();

    unsigned int nbPointsFirst = m_FirstDataset->GetNumberOfPoints();
    unsigned int nbPointsSecond = m_SecondDataset->GetNumberOfPoints();

    if (m_Verbose)
        std::cout << "Number of fiber points: ref: " << nbPointsFirst << ", moving: " << nbPointsSecond << std::endl;

    double dataSize = nbPointsFirst * nbPointsSecond * sizeof(double) / std::pow(1024.0,3);

    if (m_Verbose)
        std::cout << "Required memory to precompute distance matrix: " << dataSize << " Gb" << std::endl;

    bool precomputeDistanceMatrix = (dataSize < m_MemorySizeLimit);

    if (m_Verbose)
        std::cout << "Precomputing? " << (precomputeDistanceMatrix ? "Yes" : "No") << std::endl;

    if (precomputeDistanceMatrix)
       this->PrecomputeDistanceMatrix();
    else
        m_DistanceMatrix.set_size(0,0);

    // Now go on with the Sinkhorn algorithm
    m_UVector.resize(nbPointsFirst);
    m_VVector.resize(nbPointsSecond);

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
        for (unsigned int i = 0;i < nbPointsFirst;++i)
        {
            double diffVal = std::abs(m_OldUVector[i] - m_UVector[i]) / std::max(std::abs(m_OldUVector[i]),std::abs(m_UVector[i]));
            if (diffVal > maxDiff)
                maxDiff = diffVal;
        }

        for (unsigned int i = 0;i < nbPointsSecond;++i)
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
FiberBundlePointwiseUnbalancedOptimalTransport
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
FiberBundlePointwiseUnbalancedOptimalTransport
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

    for (unsigned int i = startIndex;i < endIndex;++i)
        tmpArg->uotPtr->ComputeExtraDataOnCell(i,0);

    step = nbTotalCellsSecond / numTotalThread;
    startIndex = nbThread * step;
    endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbTotalCellsSecond;

    for (unsigned int i = startIndex;i < endIndex;++i)
        tmpArg->uotPtr->ComputeExtraDataOnCell(i,1);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

void
FiberBundlePointwiseUnbalancedOptimalTransport
::ComputeExtraDataOnCell(unsigned int cellIndex, unsigned int dataIndex)
{
    vtkGenericCell *cellData = vtkGenericCell::New();
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
FiberBundlePointwiseUnbalancedOptimalTransport
::PrecomputeDistanceMatrix()
{
    unsigned int nbPointsFirst = m_FirstDataset->GetNumberOfPoints();
    unsigned int nbPointsSecond = m_SecondDataset->GetNumberOfPoints();

    m_DistanceMatrix.set_size(nbPointsFirst, nbPointsSecond);

    ThreadArguments tmpStr;
    tmpStr.uotPtr = this;

    this->GetMultiThreader()->SetSingleMethod(ThreadPrecomputeDistanceMatrix, &tmpStr);
    this->GetMultiThreader()->SingleMethodExecute();
}

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
FiberBundlePointwiseUnbalancedOptimalTransport
::ThreadPrecomputeDistanceMatrix(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;
    unsigned int numTotalThread = threadArgs->NumberOfWorkUnits;

    ThreadArguments *tmpArg = (ThreadArguments *)threadArgs->UserData;
    unsigned int nbTotalPointsFirst = tmpArg->uotPtr->GetFirstDataset()->GetNumberOfPoints();

    unsigned int step = nbTotalPointsFirst / numTotalThread;
    unsigned int startIndex = nbThread * step;
    unsigned int endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbTotalPointsFirst;

    tmpArg->uotPtr->PrecomputeDistancesOnRange(startIndex, endIndex);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

void
FiberBundlePointwiseUnbalancedOptimalTransport
::PrecomputeDistancesOnRange(unsigned int startIndex, unsigned int endIndex)
{
    unsigned int nbPointsSecond = m_SecondDataset->GetNumberOfPoints();

    for (unsigned int i = startIndex;i < endIndex;++i)
    {
        for (unsigned int j = 0;j < nbPointsSecond;++j)
            m_DistanceMatrix(i,j) = this->ComputeDistance(i,j);
    }
}

double
FiberBundlePointwiseUnbalancedOptimalTransport
::ComputeDistance(unsigned int firstIndex, unsigned int secondIndex)
{
    double distance = 0.0;
    double tangentDistance = 0.0;

    double firstPointPosition[3];
    double secondPointPosition[3];
    double firstTangent[3];
    double secondTangent[3];

    m_FirstDataset->GetPoint(firstIndex, firstPointPosition);
    m_SegmentTangentsFirstDataset->GetTuple(firstIndex, firstTangent);
    m_SecondDataset->GetPoint(secondIndex, secondPointPosition);
    m_SegmentTangentsSecondDataset->GetTuple(secondIndex, secondTangent);

    for (unsigned int i = 0;i < 3;++i)
    {
        distance += (firstPointPosition[i] - secondPointPosition[i]) * (firstPointPosition[i] - secondPointPosition[i]);
        tangentDistance += firstTangent[i] * secondTangent[i];
    }

    return distance * (1.0 + m_AlphaValue * (1.0 - std::pow(tangentDistance, m_KValue)));
}

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
FiberBundlePointwiseUnbalancedOptimalTransport
::ThreadVectorUpdate(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;
    unsigned int numTotalThread = threadArgs->NumberOfWorkUnits;

    ThreadArguments *tmpArg = (ThreadArguments *)threadArgs->UserData;
    unsigned int nbPoints = 0;

    if (tmpArg->uotPtr->GetUpdateUVector())
        nbPoints = tmpArg->uotPtr->GetFirstDataset()->GetNumberOfPoints();
    else
        nbPoints = tmpArg->uotPtr->GetSecondDataset()->GetNumberOfPoints();

    unsigned int step = nbPoints / numTotalThread;
    unsigned int startIndex = nbThread * step;
    unsigned int endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbPoints;

    tmpArg->uotPtr->ComputeVectorUpdateOnRange(startIndex, endIndex);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

void
FiberBundlePointwiseUnbalancedOptimalTransport
::ComputeVectorUpdateOnRange(unsigned int startIndex, unsigned int endIndex)
{
    unsigned int nbPoints = 0;
    unsigned int nbOtherPoints = 0;

    std::vector <double> *updatedVector = &m_UVector;
    std::vector <double> *fixedVector = &m_VVector;

    if (m_UpdateUVector)
    {
        nbPoints = m_FirstDataset->GetNumberOfPoints();
        nbOtherPoints = m_SecondDataset->GetNumberOfPoints();
    }
    else
    {
        nbPoints = m_SecondDataset->GetNumberOfPoints();
        nbOtherPoints = m_FirstDataset->GetNumberOfPoints();

        updatedVector = &m_VVector;
        fixedVector = &m_UVector;
    }

    std::vector <double> tmpVector(nbOtherPoints, 0.0);

    bool useDistMatrix = (m_DistanceMatrix.rows() != 0);
    bool transposeDistMatrix = (!m_UpdateUVector);
    double lambda = m_RhoValue / (m_RhoValue + m_EpsilonValue);

    for (unsigned int i = startIndex;i < endIndex;++i)
    {
        double weightValue = 1.0;
        updatedVector->operator[](i) = lambda * std::log(weightValue);

        for (unsigned int j = 0;j < nbOtherPoints;++j)
        {
            double distance = 0.0;
            if (useDistMatrix)
            {
                if (!transposeDistMatrix)
                    distance = m_DistanceMatrix(i,j);
                else
                    distance = m_DistanceMatrix(j,i);
            }
            else
            {
                if (m_UpdateUVector)
                    distance = this->ComputeDistance(i,j);
                else
                    distance = this->ComputeDistance(j,i);
            }

            tmpVector[j] = fixedVector->operator[](j) - distance / m_EpsilonValue;
        }

        updatedVector->operator[](i) -= lambda * anima::ExponentialSum(tmpVector);
    }
}

void
FiberBundlePointwiseUnbalancedOptimalTransport
::ComputeWassersteinDistanceFomData()
{
    unsigned int nbPointsFirst = m_FirstDataset->GetNumberOfPoints();
    unsigned int nbPointsSecond = m_SecondDataset->GetNumberOfPoints();

    m_WassersteinSquaredDistance = 0.0;
    std::vector <double> distKLUVector(nbPointsFirst, 0.0);
    std::vector <double> distKLVVector(nbPointsSecond, 0.0);

    bool useDistanceMatrix = (m_DistanceMatrix.rows() != 0);

    for (unsigned int i = 0;i < nbPointsFirst;++i)
    {
        for (unsigned int j = 0;j < nbPointsSecond;++j)
        {
            double ijDistance = 0.0;
            if (useDistanceMatrix)
                ijDistance = m_DistanceMatrix(i,j);
            else
                ijDistance = this->ComputeDistance(i,j);

            double Pij = std::exp(m_UVector[i] + m_VVector[j] - ijDistance / m_EpsilonValue);
            m_WassersteinSquaredDistance += Pij * ijDistance;
            distKLUVector[i] += Pij;
            distKLVVector[j] += Pij;
        }
    }

    for (unsigned int i = 0;i < nbPointsFirst;++i)
    {
        double aValue = 1.0;
        double xlnx = 0.0;
        if (distKLUVector[i] > 0)
            xlnx = distKLUVector[i] * std::log(distKLUVector[i] / aValue);

        m_WassersteinSquaredDistance += m_RhoValue * (xlnx - distKLUVector[i] + aValue);
    }

    for (unsigned int i = 0;i < nbPointsSecond;++i)
    {
        double bValue = 1.0;
        double xlnx = 0.0;
        if (distKLVVector[i] > 0)
            xlnx = distKLVVector[i] * std::log(distKLVVector[i] / bValue);

        m_WassersteinSquaredDistance += m_RhoValue * (xlnx - distKLVVector[i] + bValue);
    }
}

} // end namespace anima
