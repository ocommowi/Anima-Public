#include <animaDTIProbabilisticTractographyImageFilter.h>
#include <itkTimeProbe.h>

#include <tclap/CmdLine.h>

#include <animaReadWriteFunctions.h>
#include <animaLogTensorImageFilter.h>
#include <animaGradientFileReader.h>
#include <animaShapesWriter.h>

#include <itkCommand.h>

//Update progression of the process
void eventCallback (itk::Object* caller, const itk::EventObject& event, void* clientData)
{
    itk::ProcessObject * processObject = (itk::ProcessObject*) caller;
    std::cout<<"\033[K\rProgression: "<<(int)(processObject->GetProgress() * 100)<<"%"<<std::flush;
}

int main(int argc,  char*  argv[])
{
    TCLAP::CmdLine cmd("INRIA / IRISA - VisAGeS/Empenn Team", ' ',ANIMA_VERSION);

    // Mandatory arguments
    TCLAP::ValueArg<std::string> dtiArg("i","dti","Input diffusion tensor image",true,"","dti image",cmd);
    TCLAP::ValueArg<std::string> seedMaskArg("s","seed-mask","Seed mask",true,"","seed",cmd);
    TCLAP::ValueArg<std::string> fibersArg("o","fibers","Output fibers",true,"","fibers",cmd);
    TCLAP::ValueArg<std::string> b0Arg("b","b0","B0 image",true,"","b0 image",cmd);
    TCLAP::ValueArg<std::string> noiseArg("N","noise","Noise variance image",true,"","noise variance image",cmd);
    
    TCLAP::ValueArg<std::string> gradientsArg("g","gradients","Gradient table",true,"","gradients",cmd);
    TCLAP::ValueArg<std::string> bvaluesArg("b","bvalues","B-value list",true,"","b-values",cmd);

    // Optional arguments
    TCLAP::ValueArg<std::string> cutMaskArg("c","cut-mask","Mask for cutting fibers (default: none)",false,"","cut mask",cmd);
    TCLAP::ValueArg<std::string> forbiddenMaskArg("f","forbidden-mask","Mask for removing fibers (default: none)",false,"","remove mask",cmd);
    TCLAP::ValueArg<std::string> filterMaskArg("","filter-mask","Mask for filtering fibers (default: none)",false,"","filter mask",cmd);
    
    TCLAP::ValueArg<double> faThrArg("","fa-thr","FA threshold (default: 0.2)",false,0.2,"fa threshold",cmd);
    TCLAP::ValueArg<double> stepLengthArg("","step-length","Length of each step (default: 1)",false,1.0,"step length",cmd);
    TCLAP::ValueArg<int> nbFibersArg("","nb-fibers","Number of starting particle filters (n*n*n) per voxel (default: 1)",false,1,"number of seeds per voxel",cmd);
    
    TCLAP::ValueArg<double> minLengthArg("","min-length","Minimum length for a fiber to be considered for computation (default: 10mm)",false,10.0,"minimum length",cmd);
    TCLAP::ValueArg<double> maxLengthArg("","max-length","Maximum length of a tract (default: 150mm)",false,150.0,"maximum length",cmd);
    
    TCLAP::ValueArg<unsigned int> nbParticlesArg("n","nb-particles","Number of particles per filter (default: 1000)",false,1000,"number of particles",cmd);
    TCLAP::ValueArg<unsigned int> clusterMinSizeArg("","cluster-min-size","Minimal number of particles per cluster before split (default: 10)",false,10,"minimal cluster size",cmd);
    
    TCLAP::ValueArg<double> resampThrArg("r","resamp-thr","Resampling relative threshold (default: 0.8)",false,0.8,"resampling threshold",cmd);
    TCLAP::ValueArg<double> kappaPriorArg("k","kappa-prior","Kappa of prior distribution (default: 15)",false,15.0,"prior kappa",cmd);
    TCLAP::ValueArg<double> distThrArg("","dist-thr","Hausdorff distance threshold for mergine clusters (default: 0.5)",false,0.5,"merging threshold",cmd);
    TCLAP::ValueArg<double> kappaThrArg("","kappa-thr","Kappa threshold for splitting clusters (default: 30)",false,30.0,"splitting threshold",cmd);
    TCLAP::ValueArg<unsigned int> clusterDistArg("","cluster-dist","Distance between clusters: choices are 0 (AHD), 1 (HD, default) or 2 (MHD)",false,1,"cluster distance",cmd);
    
    TCLAP::SwitchArg averageClustersArg("M","average-clusters","Output only cluster mean",cmd,false);
    TCLAP::SwitchArg addLocalDataArg("L","local-data","Add local data information to output tracks",cmd);

    TCLAP::ValueArg<unsigned int> nbThreadsArg("T","nb-threads","Number of threads to run on (default: all available)",false,itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads(),"number of threads",cmd);
    
    try
    {
        cmd.parse(argc,argv);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    typedef anima::DTIProbabilisticTractographyImageFilter MainFilterType;
    typedef MainFilterType::InputModelImageType InputModelImageType;
    typedef MainFilterType::MaskImageType MaskImageType;
    typedef MainFilterType::Vector3DType Vector3DType;
    
    MainFilterType::Pointer dtiTracker = MainFilterType::New();
    
    // Load model image
    typedef anima::LogTensorImageFilter <double, 3> LogFilterType;

    LogFilterType::Pointer logFilter = LogFilterType::New();
    logFilter->SetInput(anima::readImage <InputModelImageType> (dtiArg.getValue()));
    logFilter->SetNumberOfWorkUnits(nbThreadsArg.getValue());

    logFilter->Update();
    dtiTracker->SetInputModelImage(logFilter->GetOutput());
    
    // Load gradient table and b-value list
    std::cout << "Importing gradient table and corresponding b-values..." << std::endl;

    typedef anima::GradientFileReader < Vector3DType, double > GFReaderType;
    GFReaderType gfReader;
    gfReader.SetGradientFileName(gradientsArg.getValue());
    gfReader.SetBValueBaseString(bvaluesArg.getValue());
    gfReader.SetGradientIndependentNormalization(false);
    gfReader.Update();

    GFReaderType::GradientVectorType directions = gfReader.GetGradients();

    for(unsigned int i = 0;i < directions.size();++i)
        dtiTracker->AddGradientDirection(i,directions[i]);

    GFReaderType::BValueVectorType mb = gfReader.GetBValues();

    dtiTracker->SetBValuesList(mb);

    // Load seed mask    
    dtiTracker->SetSeedMask(anima::readImage <MaskImageType> (seedMaskArg.getValue()));
    
    // Load cut mask
    if (cutMaskArg.getValue() != "")
        dtiTracker->SetCutMask(anima::readImage <MaskImageType> (cutMaskArg.getValue()));
    
    // Load forbidden mask
    if (forbiddenMaskArg.getValue() != "")
        dtiTracker->SetForbiddenMask(anima::readImage <MaskImageType> (forbiddenMaskArg.getValue()));
    
    // Load filter mask
    if (filterMaskArg.getValue() != "")
        dtiTracker->SetFilterMask(anima::readImage <MaskImageType> (filterMaskArg.getValue()));

    typedef MainFilterType::ScalarImageType ScalarImageType;
    dtiTracker->SetB0Image(anima::readImage <ScalarImageType> (b0Arg.getValue()));
    dtiTracker->SetNoiseImage(anima::readImage <ScalarImageType> (noiseArg.getValue()));
    
    dtiTracker->SetNumberOfFibersPerPixel(nbFibersArg.getValue());
    dtiTracker->SetStepProgression(stepLengthArg.getValue());
    dtiTracker->SetFAThreshold(faThrArg.getValue());
    dtiTracker->SetMinLengthFiber(minLengthArg.getValue());
    dtiTracker->SetMaxLengthFiber(maxLengthArg.getValue());
    
    dtiTracker->SetNumberOfParticles(nbParticlesArg.getValue());
    dtiTracker->SetMinimalNumberOfParticlesPerClass(clusterMinSizeArg.getValue());
    dtiTracker->SetResamplingThreshold(resampThrArg.getValue());
    dtiTracker->SetKappaOfPriorDistribution(kappaPriorArg.getValue());
    
    dtiTracker->SetPositionDistanceFuseThreshold(distThrArg.getValue());
    dtiTracker->SetKappaSplitThreshold(kappaThrArg.getValue());
    
    dtiTracker->SetClusterDistance(clusterDistArg.getValue());
    
    bool computeLocalColors = (fibersArg.getValue().find(".fds") != std::string::npos) && (addLocalDataArg.isSet());
    dtiTracker->SetComputeLocalColors(computeLocalColors);
    dtiTracker->SetMAPMergeFibers(averageClustersArg.isSet());
    dtiTracker->SetNumberOfWorkUnits(nbThreadsArg.getValue());

    itk::CStyleCommand::Pointer callback = itk::CStyleCommand::New();
    callback->SetCallback(eventCallback);
    dtiTracker->AddObserver(itk::ProgressEvent(), callback);

    itk::TimeProbe tmpTime;
    tmpTime.Start();
    
    dtiTracker->Update();
    dtiTracker->RemoveAllObservers();

    tmpTime.Stop();
    std::cout << "Tracking time: " << tmpTime.GetTotal() << "s" << std::endl;
    
    anima::ShapesWriter writer;
    writer.SetInputData(dtiTracker->GetOutput());
    writer.SetFileName(fibersArg.getValue());
    writer.Update();
    
    return EXIT_SUCCESS;
}
