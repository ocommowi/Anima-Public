#include <tclap/CmdLine.h>

#include <animaShapesReader.h>
#include <animaFiberBundleUnbalancedOptimalTransport.h>
#include <animaFiberBundlePointwiseUnbalancedOptimalTransport.h>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkGenericCell.h>

#include <itkPoolMultiThreader.h>
#include <itkTimeProbe.h>

int main(int argc, char **argv)
{
    TCLAP::CmdLine cmd("INRIA / IRISA - VisAGeS/Empenn Team", ' ',ANIMA_VERSION);

    TCLAP::ValueArg <std::string> refArg("r","ref","Fibers set to be compared",true,"","fibers",cmd);
    TCLAP::ValueArg <std::string> movArg("m","moving","Fibers set to be compared",true,"","fibers",cmd);

    TCLAP::ValueArg <unsigned int> precisionArg("p","precision","Precision of values output (integer, default: 6)",false,6,"precision",cmd);
    TCLAP::ValueArg <unsigned int> nbThreadsArg("T","nb-threads","Number of threads to run on (default: all available)",false,itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads(),"number of threads",cmd);

    TCLAP::ValueArg <double> rhoArg("R","rho","Rho value in unbalanced optimal transport (default: 100.0)",false,100.0,"rho value in UOT",cmd);
    TCLAP::ValueArg <double> epsilonArg("E","epsilon","Epsilon value in unbalanced optimal transport (default: sqrt(0.07))",false,std::sqrt(0.07),"epsilon value in UOT",cmd);

    TCLAP::ValueArg <double> stopThrArg("s","stop-thr","Relative threshold to stop iterations (default: 1.0e-4)",false,1.0e-4,"relative stop threshold",cmd);
    TCLAP::ValueArg <double> memoryLimitArg("M","mem-lim","Memory limit to precompute distance matrix (in Gb, default: 8)",false,8.0,"memory limit",cmd);

    TCLAP::ValueArg <unsigned int> distTypeArg("d","dist-type","distance type (0: point-wise distance, 1: fiber-wise distance, default: 1)",false,1,"distance type",cmd);

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

    // Get dummy cell so that it's thread safe
    vtkSmartPointer <vtkGenericCell> dummyCell = vtkGenericCell::New();
    refDataTracks->GetCell(0,dummyCell);

    vtkIdType nbTotalPtsRef = refDataTracks->GetNumberOfPoints();
    if (nbTotalPtsRef == 0)
    {
        std::cout << "No points in reference track file, nothing to do" << std::endl;
        return EXIT_FAILURE;
    }

    // Get dummy cell so that it's thread safe
    dummyCell = vtkGenericCell::New();
    refDataTracks->GetCell(0,dummyCell);

    // Now go for moving tracks, again compute tangents
    trackReader.SetFileName(movArg.getValue());
    trackReader.Update();

    PolyDataPointer movingDataTracks = trackReader.GetOutput();
    dummyCell = vtkGenericCell::New();
    movingDataTracks->GetCell(0,dummyCell);

    vtkIdType nbTotalPtsMoving = movingDataTracks->GetNumberOfPoints();
    if (nbTotalPtsMoving == 0)
    {
        std::cout << "No points in moving track file, nothing to do" << std::endl;
        return EXIT_FAILURE;
    }

    // Get dummy cell so that it's thread safe
    movingDataTracks->GetCell(0,dummyCell);

    double distanceValue = 0;
    if (distTypeArg.getValue() == 0) // pointwise distance
    {
        anima::FiberBundlePointwiseUnbalancedOptimalTransport::Pointer uotFilter = anima::FiberBundlePointwiseUnbalancedOptimalTransport::New();
        uotFilter->SetFirstDataset(refDataTracks);
        uotFilter->SetSecondDataset(movingDataTracks);
        uotFilter->SetAlphaValue(1.0);
        uotFilter->SetKValue(4.0);
        uotFilter->SetEpsilonValue(epsilonArg.getValue());
        uotFilter->SetRhoValue(rhoArg.getValue());
        uotFilter->SetMemorySizeLimit(memoryLimitArg.getValue());
        uotFilter->SetRelativeStopCriterion(stopThrArg.getValue());
        uotFilter->SetNumberOfWorkUnits(nbThreadsArg.getValue());
        uotFilter->SetVerbose(true);

        uotFilter->Update();
        distanceValue = uotFilter->GetWassersteinSquaredDistance();
    }
    else // fiber wise distance
    {
        anima::FiberBundleUnbalancedOptimalTransport::Pointer uotFilter = anima::FiberBundleUnbalancedOptimalTransport::New();
        uotFilter->SetFirstDataset(refDataTracks);
        uotFilter->SetSecondDataset(movingDataTracks);
        uotFilter->SetAlphaValue(1.0);
        uotFilter->SetKValue(4.0);
        uotFilter->SetEpsilonValue(epsilonArg.getValue());
        uotFilter->SetRhoValue(rhoArg.getValue());
        uotFilter->SetMemorySizeLimit(memoryLimitArg.getValue());
        uotFilter->SetRelativeStopCriterion(stopThrArg.getValue());
        uotFilter->SetNumberOfWorkUnits(nbThreadsArg.getValue());
        uotFilter->SetVerbose(true);

        uotFilter->Update();
        distanceValue = uotFilter->GetWassersteinSquaredDistance();
    }

    std::cout.precision(precisionArg.getValue());
    std::cout << "Squared Wasserstein distance: " << distanceValue << std::endl;

    return EXIT_SUCCESS;
}
