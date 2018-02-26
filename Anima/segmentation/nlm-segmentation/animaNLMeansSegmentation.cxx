#include <iostream>
#include <tclap/CmdLine.h>

#include <animaReadWriteFunctions.h>
#include <animaNLMeansSegmentationImageFilter.h>
#include <itkTimeProbe.h>

//Update progression of the process
void eventCallback (itk::Object* caller, const itk::EventObject& event, void* clientData)
{
    itk::ProcessObject * processObject = (itk::ProcessObject*) caller;
    std::cout<<"\033[K\rProgression: "<<(int)(processObject->GetProgress() * 100)<<"%"<<std::flush;
}

int main(int argc, char **argv)
{
    TCLAP::CmdLine cmd("INRIA / IRISA - Visages Team", ' ',ANIMA_VERSION);
    
    TCLAP::ValueArg<std::string> refArg("i","input","Image to delineate",true,"","image to delineate",cmd);
    TCLAP::ValueArg<std::string> dataArg("I","database","Database image list",true,"","database image list",cmd);
    TCLAP::ValueArg<std::string> dataSegArg("S","database-seg","Segmentations database image list",true,"","segmentation database image list",cmd);

    TCLAP::ValueArg<std::string> maskArg("m","maskname","Computation mask",false,"","computation mask",cmd);
    TCLAP::ValueArg<std::string> resArg("o","output","Labeled image",true,"","labeled image",cmd);

    TCLAP::ValueArg<double> weightThrArg("w","weightthr","NL weight threshold: patches around have to be similar enough (default: 0.0)",false,0.0,"NL weight threshold",cmd);
    TCLAP::ValueArg<double> thrArg("t","thr","Tolerance for keeping patch in test, default: 0.95)",false,0.95,"patch tolerance",cmd);
    
    TCLAP::ValueArg<unsigned int> nbpArg("p","numberofthreads","Number of threads to run on (default: all cores)",false,itk::MultiThreader::GetGlobalDefaultNumberOfThreads(),"number of threads",cmd);

    TCLAP::ValueArg<unsigned int> patchHSArg("","patchhalfsize","Patch half size in each direction (default: 1)",false,1,"patch half size",cmd);
    TCLAP::ValueArg<unsigned int> patchSSArg("","patchstepsize","Patch step size for searching (default: 1)",false,1,"patch search step size",cmd);
    TCLAP::ValueArg<unsigned int> patchNeighArg("n","patchneighborhood","Patch half neighborhood size (default: 3)",false,3,"patch search neighborhood size",cmd);
    
    try
    {
        cmd.parse(argc,argv);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }
    
    typedef anima::NLMeansSegmentationImageFilter <double, unsigned short> NLSegmentationImageFilterType;
    typedef NLSegmentationImageFilterType::InputImageType ImageType;
    typedef NLSegmentationImageFilterType::OutputImageType SegmentationImageType;

    itk::CStyleCommand::Pointer callback = itk::CStyleCommand::New();
    callback->SetCallback(eventCallback);

    NLSegmentationImageFilterType::Pointer mainFilter = NLSegmentationImageFilterType::New();

    if (maskArg.getValue() != "")
        mainFilter->SetComputationMask(anima::readImage < itk::Image <unsigned char, 3> > (maskArg.getValue()));

    mainFilter->SetNumberOfThreads(nbpArg.getValue());

    mainFilter->SetWeightThreshold(weightThrArg.getValue());
    mainFilter->SetThreshold(thrArg.getValue());

    mainFilter->SetPatchHalfSize(patchHSArg.getValue());
    mainFilter->SetSearchStepSize(patchSSArg.getValue());
    mainFilter->SetSearchNeighborhood(patchNeighArg.getValue());

    mainFilter->SetInput(0,anima::readImage <ImageType> (refArg.getValue()));
    mainFilter->AddObserver(itk::ProgressEvent(), callback);

    std::ifstream fileIn(dataArg.getValue());
    
    while (!fileIn.eof())
    {
        char tmpStr[2048];
        fileIn.getline(tmpStr,2048);
        
        if (strcmp(tmpStr,"") == 0)
            continue;
        
        std::cout << "Loading database image " << tmpStr << "..." << std::endl;
        mainFilter->AddDatabaseInput(anima::readImage <ImageType> (tmpStr));
    }
    fileIn.close();
    
    std::ifstream fileSegIn(dataSegArg.getValue());

    while (!fileSegIn.eof())
    {
        char tmpStr[2048];
        fileSegIn.getline(tmpStr,2048);

        if (strcmp(tmpStr,"") == 0)
            continue;

        std::cout << "Loading database segmentation image " << tmpStr << "..." << std::endl;
        mainFilter->AddDatabaseSegmentationInput(anima::readImage <SegmentationImageType> (tmpStr));
    }
    fileSegIn.close();

    itk::TimeProbe tmpTime;
    tmpTime.Start();

    try
    {
        mainFilter->Update();
    }
    catch (itk::ExceptionObject &e)
    {
        std::cerr << e << std::endl;
        return EXIT_FAILURE;
    }
    
    tmpTime.Stop();
    std::cout << "\nSegmentation time: " << tmpTime.GetTotal() << "s..." << std::endl;

    anima::writeImage <SegmentationImageType> (resArg.getValue(),mainFilter->GetOutput());

    return EXIT_SUCCESS;
}
