#include <tclap/CmdLine.h>

#include <animaReadWriteFunctions.h>
#include <itkImage.h>
#include <itkVectorImage.h>
#include <itkTimeProbe.h>

#include <itkGradientImageFilter.h>
#include <animaStructureTensorImageFilter.h>

int main(int argc, char **argv)
{
    std::string descriptionMessage = "Computes structure tensors image from a scalar image\n";
    descriptionMessage += "INRIA / IRISA - VisAGeS Team";

    TCLAP::CmdLine cmd(descriptionMessage, ' ',ANIMA_VERSION);

    TCLAP::ValueArg<std::string> inArg("i","input","Input image",true,"","input image",cmd);
    TCLAP::ValueArg<std::string> outArg("o","outputfile","output image",true,"","output image",cmd);

    TCLAP::SwitchArg realCoordsArg("R","real","Compute structure tensors in real coordinates (default: voxel ones)",cmd);
    TCLAP::SwitchArg normalizeArg("N","normalize","Normalize structure tensors",cmd);
    TCLAP::ValueArg<unsigned int> neighArg("n","neigh","Neighborhood size for tensor computation (default: 1)",false,1,"neighborhood size",cmd);

    TCLAP::ValueArg <unsigned int> nbpArg("T","numberofthreads","Number of threads to run on (default : all cores)",false,itk::MultiThreader::GetGlobalDefaultNumberOfThreads(),"number of threads",cmd);

    try
    {
        cmd.parse(argc,argv);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    typedef itk::Image <float,3> ImageType;
    typedef itk::GradientImageFilter <ImageType,double,double> GradientFilterType;
    typedef itk::VectorImage <double,3> VectorImageType;

    itk::TimeProbe tmpTimer;
    tmpTimer.Start();

    GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
    gradientFilter->SetInput(anima::readImage <ImageType> (inArg.getValue()));
    gradientFilter->SetNumberOfThreads(nbpArg.getValue());

    if (realCoordsArg.isSet())
    {
        gradientFilter->UseImageDirectionOn();
        gradientFilter->UseImageSpacingOn();
    }
    else
    {
        gradientFilter->UseImageDirectionOff();
        gradientFilter->UseImageSpacingOff();
    }

    gradientFilter->Update();

    typedef anima::StructureTensorImageFilter <GradientFilterType::OutputImageType::PixelType,double,3> StructureTensorFilterType;

    StructureTensorFilterType::Pointer structureFilter = StructureTensorFilterType::New();
    structureFilter->SetInput(gradientFilter->GetOutput());
    structureFilter->SetNumberOfThreads(nbpArg.getValue());
    structureFilter->SetNeighborhood(std::max((unsigned int)1,neighArg.getValue()));
    structureFilter->SetNormalize(normalizeArg.isSet());

    structureFilter->Update();

    tmpTimer.Stop();
    std::cout << "Execution time: " << tmpTimer.GetTotal() << "s" << std::endl;

    anima::writeImage <VectorImageType> (outArg.getValue(),structureFilter->GetOutput());

    return EXIT_SUCCESS;
}
