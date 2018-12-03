#include <animaMeanAndVarianceImagesFilter.h>
#include <animaReadWriteFunctions.h>

#include <tclap/CmdLine.h>

int main(int argc, char **argv)
{
    TCLAP::CmdLine cmd("INRIA / IRISA - VisAGeS Team", ' ',ANIMA_VERSION);

    TCLAP::ValueArg<std::string> inArg("i","input","Input image",true,"","input image",cmd);
    TCLAP::ValueArg<std::string> resMeanArg("o","out-mean","Result local mean image",true,"","result local mean image",cmd);
    TCLAP::ValueArg<std::string> resVarArg("O","out-var","Result local variance image",true,"","result local variance image",cmd);

    TCLAP::ValueArg<unsigned int> neighArg("n","neigh-size","Neighborhood half size (default: 2)",false,2,"half size of local patch",cmd);
    TCLAP::ValueArg<unsigned int> nbpArg("T","nb-threads","Number of threads (default: all cores)",false,itk::MultiThreader::GetGlobalDefaultNumberOfThreads(),"Number of threads",cmd);

    try
    {
        cmd.parse(argc,argv);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    typedef itk::Image <double, 3> ImageType;
    typedef anima::MeanAndVarianceImagesFilter <ImageType, ImageType> MainFilterType;

    MainFilterType::Pointer mainFilter = MainFilterType::New();

    mainFilter->SetNumberOfThreads(nbpArg.getValue());

    ImageType::SizeType radius;
    for (unsigned int i = 0;i < 3;++i)
    {
        radius[i] = neighArg.getValue();
    }
    mainFilter->SetRadius(radius);

    mainFilter->SetInput(anima::readImage <ImageType> (inArg.getValue()));

    mainFilter->Update();

    std::cout << "Writing result to : " << resMeanArg.getValue() << std::endl;
    anima::writeImage <ImageType> (resMeanArg.getValue(),mainFilter->GetMeanImage());

    std::cout << "Writing result to : " << resVarArg.getValue() << std::endl;
    anima::writeImage <ImageType> (resVarArg.getValue(),mainFilter->GetVarImage());

    return EXIT_SUCCESS;
}
