#include <tclap/CmdLine.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTransformFileReader.h>
#include <itkAffineTransform.h>
#include <itkChangeInformationImageFilter.h>

#include <itkImage.h>
#include <itkOrientImageFilter.h>
#include <animaReorientation.h>
#include <animaReadWriteFunctions.h>
#include <animaRetrieveImageTypeMacros.h>

int main(int argc, char **argv)
{
    std::string descriptionMessage = "Apply a similarity transform (rigid o isotropic scaling) without resampling by modifying image orientation";
    descriptionMessage += "INRIA / IRISA - Visages Team";

    TCLAP::CmdLine cmd(descriptionMessage, ' ', ANIMA_VERSION);

    TCLAP::ValueArg<std::string> inArg("i", "input", "input_filename", true, "", "input image.", cmd);
    TCLAP::ValueArg<std::string> trArg("t", "trsf", "transform", true, "", "transform", cmd);
    TCLAP::ValueArg<std::string> outArg("o", "output", "Output image", true, "", "output image", cmd);

    try
    {
        cmd.parse(argc, argv);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return(1);
    }

    // Find out the type of the image in file
    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(inArg.getValue().c_str(),
        itk::ImageIOFactory::ReadMode);

    if (!imageIO)
    {
        std::cerr << "Itk could not find suitable IO factory for the input" << std::endl;
        return EXIT_FAILURE;
    }

    // Now that we found the appropriate ImageIO class, ask it to read the meta data from the image file.
    imageIO->SetFileName(inArg.getValue());
    imageIO->ReadImageInformation();

    typedef itk::ImageIOBase::IOPixelType     IOPixelType;
    const IOPixelType pixelType = imageIO->GetPixelType();
    std::cout << "Pixel Type is " << itk::ImageIOBase::GetPixelTypeAsString(pixelType) << std::endl;
    typedef itk::ImageIOBase::IOComponentType IOComponentType;
    const IOComponentType componentType = imageIO->GetComponentType();
    std::cout << "Component Type is " << imageIO->GetComponentTypeAsString(componentType) << std::endl;
    const unsigned int nbDimension = imageIO->GetNumberOfDimensions();
    std::cout << "Image Dimension is " << nbDimension << std::endl << std::endl;

    const unsigned int Dimension = 3;

    using PixelType = float;
    using ImageType = itk::Image< PixelType, Dimension >;

    using ReaderType = itk::ImageFileReader< ImageType >;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(inArg.getValue());
    try
    {
        reader->UpdateOutputInformation();
    }
    catch (itk::ExceptionObject & error)
    {
        std::cerr << "Error: " << error << std::endl;
        return EXIT_FAILURE;
    }
    ImageType::ConstPointer inputImage = reader->GetOutput();

    using FilterType = itk::ChangeInformationImageFilter< ImageType >;
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput(reader->GetOutput());
    
    using PrecisionType = double;
    using MatrixTransformType = itk::AffineTransform <PrecisionType, Dimension>;
    typedef itk::AffineTransform <PrecisionType, Dimension> MatrixTransformType;
    typedef MatrixTransformType::Pointer MatrixTransformPointer;
    typedef vnl_matrix <double> MatrixType;
    itk::TransformFileReader::Pointer trReader = itk::TransformFileReader::New();
    trReader->SetFileName(trArg.getValue());
        try
    {
        trReader->Update();
    }
    catch (itk::ExceptionObject &e)
    {
        std::cerr << "Problem reading transform file " << trArg.getValue() << ", exiting" << std::endl;
        return EXIT_FAILURE;
    }
    itk::TransformFileReader::TransformListType trsfList = *(trReader->GetTransformList());
    itk::TransformFileReader::TransformListType::iterator tr_it = trsfList.begin();
    MatrixTransformType *trsf = dynamic_cast <MatrixTransformType *> ((*tr_it).GetPointer());

    if (trsf == NULL)
    {
        std::cerr << "Problem converting transform file to linear file " << trArg.getValue() << ", exiting" << std::endl;
        return EXIT_FAILURE;
    }

    const ImageType::DirectionType oldDirection = inputImage->GetDirection();
    vnl_matrix<double> tmpDirection = trsf->GetInverseMatrix().GetVnlMatrix();
    const ImageType::DirectionType newDirection = tmpDirection.normalize_rows()*oldDirection.GetVnlMatrix();
    std::cout << std::endl << "old direction: " << std::endl << oldDirection << std::endl;
    std::cout << "new direction: " << std::endl << newDirection << std::endl;
    filter->SetOutputDirection(newDirection);
    filter->ChangeDirectionOn();

    ImageType::PointType oldOrigin = inputImage->GetOrigin();
    ImageType::PointType newOrigin = trsf->GetInverseMatrix()*(oldOrigin - trsf->GetOffset());
    std::cout << "old origin: " << std::endl << oldOrigin << std::endl;
    std::cout << std::endl << "new origin: " << std::endl << newOrigin << std::endl;
    filter->SetOutputOrigin(newOrigin);
    filter->ChangeOriginOn();

    const ImageType::SpacingType oldSpacing = inputImage->GetSpacing();
    const ImageType::SpacingType newSpacing = oldSpacing*std::sqrt(std::pow(tmpDirection.fro_norm(), 2) / Dimension);

    std::cout << std::endl << "old spacing: " << std::endl << oldSpacing << std::endl;
    std::cout << std::endl << "new spacing: " << std::endl << newSpacing << std::endl;
    filter->SetOutputSpacing(newSpacing);
    filter->ChangeSpacingOn();

    try
    {
        filter->UpdateOutputInformation();
    }
    catch (itk::ExceptionObject & error)
    {
        std::cerr << "Error: " << error << std::endl;
        return EXIT_FAILURE;
    }

    ImageType::ConstPointer output = filter->GetOutput();
    using WriterType = itk::ImageFileWriter< ImageType >;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(outArg.getValue());
    writer->SetInput(output);

    try
    {
        writer->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
        std::cerr << excp << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}