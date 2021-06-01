#include <tclap/CmdLine.h>

#include <animaShapesReader.h>
#include <animaVectorOperations.h>

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkGenericCell.h>
#include <vtkDoubleArray.h>
#include <vtkGenericCell.h>

#include <itkPoolMultiThreader.h>
#include <itkTimeProbe.h>

void ComputeTangentsOnOneCell(vtkCell *cell, vtkSmartPointer <vtkDoubleArray> &lengthParameters,
                              vtkSmartPointer <vtkDoubleArray> &tangentParameters)
{
    vtkPoints *cellPts = cell->GetPoints();
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

        int ptId = cell->GetPointId(j);

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

        tangentParameters->SetTuple3(ptId, tangent[0], tangent[1], tangent[2]);
        lengthParameters->SetValue(ptId,norm);
    }
}

double ComputePointDistance(double *firstPointPosition, double *firstTangent, double *secondPointPosition, double *secondTangent,
                            double alpha, double k)
{
    double distance = 0.0;
    double tangentDistance = 0.0;

    for (unsigned int i = 0;i < 3;++i)
    {
        distance += (firstPointPosition[i] - secondPointPosition[i]) * (firstPointPosition[i] - secondPointPosition[i]);
        tangentDistance += firstTangent[i] * secondTangent[i];
    }

    return distance * (1.0 + alpha * (1.0 - std::pow(tangentDistance, k)));
}

typedef struct
{
    vtkSmartPointer <vtkPolyData> tracks;
    vtkSmartPointer <vtkDoubleArray> lengthParameters;
    vtkSmartPointer <vtkDoubleArray> tangentParameters;
} TangentsThreaderArguments;

typedef struct
{
    std::vector <double> *currentVector;
    std::vector <double> *currentOtherVector;
    std::vector <double> *newVector;
    double epsilon, lambda;
    vtkSmartPointer <vtkPolyData> refTracks, movingTracks;
    vtkSmartPointer <vtkDoubleArray> refLengthParameters, movingLengthParameters;
    vtkSmartPointer <vtkDoubleArray> refTangentParameters, movingTangentParameters;
} VectorUpdateThreaderArguments;

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION ThreadTangent(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;
    unsigned int numTotalThread = threadArgs->NumberOfWorkUnits;

    TangentsThreaderArguments *tmpArg = (TangentsThreaderArguments *)threadArgs->UserData;
    unsigned int nbTotalCells = tmpArg->tracks->GetNumberOfCells();

    unsigned int step = nbTotalCells / numTotalThread;
    unsigned int startIndex = nbThread * step;
    unsigned int endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbTotalCells;

    vtkSmartPointer <vtkGenericCell> cell = vtkGenericCell::New();
    for (unsigned int i = startIndex;i < endIndex;++i)
    {
        tmpArg->tracks->GetCell(i,cell);
        ComputeTangentsOnOneCell(cell, tmpArg->lengthParameters, tmpArg->tangentParameters);
    }

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

void ComputeVectorUpdateOnRange(unsigned int startIndex, unsigned int endIndex,
                                VectorUpdateThreaderArguments *dataStr)
{
    unsigned int nbTotalPtsMoving = dataStr->movingTracks->GetNumberOfPoints();
    std::vector <double> tmpVector(nbTotalPtsMoving, 0.0);

    for (unsigned int i = startIndex;i < endIndex;++i)
    {
        dataStr->newVector->operator[](i) = dataStr->lambda * dataStr->currentVector->operator[](i) + dataStr->epsilon * dataStr->lambda * std::log(dataStr->refLengthParameters->GetValue(i));
        double uPosition[3];
        double uTangent[3];
        double vPosition[3];
        double vTangent[3];

        dataStr->refTracks->GetPoints()->GetPoint(i,uPosition);
        dataStr->refTangentParameters->GetTuple(i,uTangent);

        for (unsigned int j = 0;j < nbTotalPtsMoving;++j)
        {
            dataStr->movingTracks->GetPoints()->GetPoint(j,vPosition);
            dataStr->movingTangentParameters->GetTuple(j,vTangent);

            tmpVector[j] = (dataStr->currentVector->operator[](i) + dataStr->currentOtherVector->operator[](j) - ComputePointDistance(uPosition,uTangent,vPosition,vTangent,1.0,4.0)) / dataStr->epsilon;
        }

        dataStr->newVector->operator[](i) -= dataStr->epsilon * dataStr->lambda * anima::ExponentialSum(tmpVector);
    }
}

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION ThreadVectorUpdate(void *arg)
{
    itk::MultiThreaderBase::WorkUnitInfo *threadArgs = (itk::MultiThreaderBase::WorkUnitInfo *)arg;
    unsigned int nbThread = threadArgs->WorkUnitID;
    unsigned int numTotalThread = threadArgs->NumberOfWorkUnits;

    VectorUpdateThreaderArguments *tmpArg = (VectorUpdateThreaderArguments *)threadArgs->UserData;
    unsigned int nbTotalPts = tmpArg->refTracks->GetNumberOfPoints();

    unsigned int step = nbTotalPts / numTotalThread;
    unsigned int startIndex = nbThread * step;
    unsigned int endIndex = (nbThread + 1) * step;

    if (nbThread == numTotalThread - 1)
        endIndex = nbTotalPts;

    ComputeVectorUpdateOnRange(startIndex, endIndex, tmpArg);

    return ITK_THREAD_RETURN_DEFAULT_VALUE;
}

int main(int argc, char **argv)
{
    TCLAP::CmdLine cmd("INRIA / IRISA - VisAGeS/Empenn Team", ' ',ANIMA_VERSION);

    TCLAP::ValueArg <std::string> refArg("r","ref","Fibers set to be compared",true,"","fibers",cmd);
    TCLAP::ValueArg <std::string> movArg("m","moving","Fibers set to be compared",true,"","fibers",cmd);

    TCLAP::ValueArg <unsigned int> precisionArg("p","precision","Precision of values output (integer, default: 6)",false,6,"precision",cmd);
    TCLAP::ValueArg <unsigned int> nbThreadsArg("T","nb-threads","Number of threads to run on (default: all available)",false,itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads(),"number of threads",cmd);

    TCLAP::ValueArg <double> rhoArg("R","rho","Rho value in unbalanced optimal transport (default: )",false,0.25,"number of threads",cmd);
    TCLAP::ValueArg <double> epsilonArg("E","epsilon","Epsilon value in unbalanced optimal transport (default: )",false,2.25e-4,"number of threads",cmd);

    try
    {
        cmd.parse(argc,argv);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    // Get reference tracks and compute tangents
    anima::ShapesReader trackReader;
    trackReader.SetFileName(refArg.getValue());
    trackReader.Update();

    using PolyDataPointer = vtkSmartPointer <vtkPolyData>;
    PolyDataPointer refDataTracks = trackReader.GetOutput();

    vtkIdType nbTotalPtsRef = refDataTracks->GetNumberOfPoints();
    if (nbTotalPtsRef == 0)
    {
        std::cout << "No points in reference track file, nothing to do" << std::endl;
        return EXIT_FAILURE;
    }

    // Get dummy cell so that it's thread safe
    vtkSmartPointer <vtkGenericCell> dummyCell = vtkGenericCell::New();
    refDataTracks->GetCell(0,dummyCell);

    vtkSmartPointer <vtkDoubleArray> refTangents = vtkDoubleArray::New();
    refTangents->SetNumberOfComponents(3);
    refTangents->SetNumberOfTuples(nbTotalPtsRef);
    refTangents->SetName("Tangents");

    vtkSmartPointer <vtkDoubleArray> refLengths = vtkDoubleArray::New();
    refLengths->SetNumberOfComponents(1);
    refLengths->SetNumberOfTuples(nbTotalPtsRef);
    refLengths->SetName("Lengths");

    TangentsThreaderArguments tmpStr;
    tmpStr.tangentParameters = refTangents;
    tmpStr.lengthParameters = refLengths;
    tmpStr.tracks = refDataTracks;

    itk::PoolMultiThreader::Pointer mThreader = itk::PoolMultiThreader::New();
    mThreader->SetNumberOfWorkUnits(nbThreadsArg.getValue());
    mThreader->SetSingleMethod(ThreadTangent,&tmpStr);
    mThreader->SingleMethodExecute();

    refDataTracks->GetPointData()->AddArray(refLengths);
    refDataTracks->GetPointData()->AddArray(refTangents);

    // Now go for moving tracks, again compute tangents
    trackReader.SetFileName(movArg.getValue());
    trackReader.Update();

    PolyDataPointer movingDataTracks = trackReader.GetOutput();

    vtkIdType nbTotalPtsMoving = movingDataTracks->GetNumberOfPoints();
    if (nbTotalPtsMoving == 0)
    {
        std::cout << "No points in moving track file, nothing to do" << std::endl;
        return EXIT_FAILURE;
    }

    // Get dummy cell so that it's thread safe
    movingDataTracks->GetCell(0,dummyCell);

    vtkSmartPointer <vtkDoubleArray> movingTangents = vtkDoubleArray::New();
    movingTangents->SetNumberOfComponents(3);
    movingTangents->SetNumberOfTuples(nbTotalPtsMoving);
    movingTangents->SetName("Tangents");

    vtkSmartPointer <vtkDoubleArray> movingLengths = vtkDoubleArray::New();
    movingLengths->SetNumberOfComponents(1);
    movingLengths->SetNumberOfTuples(nbTotalPtsMoving);
    movingLengths->SetName("Lengths");

    tmpStr.tangentParameters = movingTangents;
    tmpStr.lengthParameters = movingLengths;
    tmpStr.tracks = movingDataTracks;

    mThreader->SetSingleMethod(ThreadTangent,&tmpStr);
    mThreader->SingleMethodExecute();

    movingDataTracks->GetPointData()->AddArray(movingLengths);
    movingDataTracks->GetPointData()->AddArray(movingTangents);

    // Now go on with the Sinkhorn algorithm
    std::vector <double> uVector(nbTotalPtsRef, 0.0), vVector(nbTotalPtsMoving, 0.0);
    std::vector <double> newUVector(nbTotalPtsRef, 0.0), newVVector(nbTotalPtsMoving, 0.0);

    double rho = rhoArg.getValue();
    double epsilon = epsilonArg.getValue();
    double lambda = rho / (rho + epsilon);

    for (unsigned int a = 0;a < 5000;++a)
    {
        itk::TimeProbe tmpTime;
        tmpTime.Start();

        VectorUpdateThreaderArguments uVecUpdateStr;
        uVecUpdateStr.refTracks = refDataTracks;
        uVecUpdateStr.refLengthParameters = refLengths;
        uVecUpdateStr.refTangentParameters = refTangents;
        uVecUpdateStr.movingTracks = movingDataTracks;
        uVecUpdateStr.movingLengthParameters = movingLengths;
        uVecUpdateStr.movingTangentParameters = movingTangents;
        uVecUpdateStr.currentVector = &uVector;
        uVecUpdateStr.currentOtherVector = &vVector;
        uVecUpdateStr.newVector = &newUVector;
        uVecUpdateStr.epsilon = epsilon;
        uVecUpdateStr.lambda = lambda;

        mThreader->SetSingleMethod(ThreadVectorUpdate,&uVecUpdateStr);
        mThreader->SingleMethodExecute();

        VectorUpdateThreaderArguments vVecUpdateStr;
        vVecUpdateStr.refTracks = movingDataTracks;
        vVecUpdateStr.refLengthParameters = movingLengths;
        vVecUpdateStr.refTangentParameters = movingTangents;
        vVecUpdateStr.movingTracks = refDataTracks;
        vVecUpdateStr.movingLengthParameters = refLengths;
        vVecUpdateStr.movingTangentParameters = refTangents;
        vVecUpdateStr.currentVector = &vVector;
        vVecUpdateStr.currentOtherVector = &uVector;
        vVecUpdateStr.newVector = &newVVector;
        vVecUpdateStr.epsilon = epsilon;
        vVecUpdateStr.lambda = lambda;

        mThreader->SetSingleMethod(ThreadVectorUpdate,&vVecUpdateStr);
        mThreader->SingleMethodExecute();

        tmpTime.Stop();

        double maxDiff = 0.0;
        for (unsigned int i = 0;i < nbTotalPtsRef;++i)
        {
            double diffVal = std::abs(newUVector[i] - uVector[i]) / std::max(std::abs(newUVector[i]),std::abs(uVector[i]));
            if (diffVal > maxDiff)
                maxDiff = diffVal;

            uVector[i] = newUVector[i];
        }

        for (unsigned int i = 0;i < nbTotalPtsMoving;++i)
        {
            double diffVal = std::abs(newVVector[i] - vVector[i]) / std::max(std::abs(newVVector[i]),std::abs(vVector[i]));
            if (diffVal > maxDiff)
                maxDiff = diffVal;

            vVector[i] = newVVector[i];
        }

        std::cout << "Stop criterion: " << maxDiff << std::endl;
        std::cout << "Iteration " << a+1 << " in " << tmpTime.GetTotal() << std::endl;
    }

    return EXIT_SUCCESS;
}
