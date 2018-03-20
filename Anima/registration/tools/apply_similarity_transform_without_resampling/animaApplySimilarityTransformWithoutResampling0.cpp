#include <tclap/CmdLine.h>

#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>
#include <itkAffineTransform.h>
#include <itkImage.h>
#include <itkOrientImageFilter.h>
#include <animaReorientation.h>
#include <animaReadWriteFunctions.h>
#include <animaRetrieveImageTypeMacros.h>
#include <vnl/vnl_inverse.h>

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
    std::cout << "Pixel Type is "
        << itk::ImageIOBase::GetPixelTypeAsString(pixelType)
        << std::endl;
    typedef itk::ImageIOBase::IOComponentType IOComponentType;
    const IOComponentType componentType = imageIO->GetComponentType();
    std::cout << "Component Type is "
        << imageIO->GetComponentTypeAsString(componentType)
        << std::endl;
    const unsigned int nbDimension = imageIO->GetNumberOfDimensions();
    std::cout << "Image Dimension is " << nbDimension << std::endl;

    typedef double PrecisionType;
    const unsigned int Dimension = 3;
    typedef itk::AffineTransform <PrecisionType, Dimension> MatrixTransformType;
    typedef MatrixTransformType::Pointer MatrixTransformPointer;
    typedef vnl_matrix <double> MatrixType;
    typedef itk::Image<IOComponentType, Dimension> IOImageType;

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

    MatrixType workMatrix(Dimension + 1, Dimension + 1), orientationMatrix(Dimension + 1, Dimension + 1);
    workMatrix.set_identity();
    orientationMatrix.set_identity();

    for (unsigned int i = 0; i < Dimension; ++i)
    {
        workMatrix(i, Dimension) = trsf->GetOffset()[i];
        for (unsigned int j = 0; j < Dimension; ++j)
            workMatrix(i, j) = trsf->GetMatrix()(i, j);
    }
    for (unsigned int i = 0; i < Dimension; ++i)
    {
        orientationMatrix(i, Dimension) = imageIO->GetOrigin(i);
        for (int j = 0; j < Dimension; ++j)
            orientationMatrix(i, j) = imageIO->GetDirection(j)[i] * imageIO->GetSpacing(j);
    }

    std::cout << std::endl << "transform matrix" << std::endl << workMatrix << std::endl;
    std::cout << "orientation matrix" << std::endl << orientationMatrix << std::endl;

    MatrixType newOrientationMatrix = vnl_inverse(workMatrix)*orientationMatrix;
    std::cout << "new orientation matrix" << std::endl << newOrientationMatrix << std::endl;

    for (unsigned int i = 0; i < Dimension; ++i)
    {
        imageIO->SetSpacing(i, newOrientationMatrix.get_column(i).magnitude());
        imageIO->SetOrigin(i, newOrientationMatrix(i, Dimension));
        imageIO->SetDirection(i, newOrientationMatrix.get_row(i).normalize());
    }

    anima::writeImage <IOImageType>(outArg.getValue(), imageIO);
    //itk::ImageFileWriter<IOImageType>::Pointer writer = itk::ImageFileWriter<IOImageType>::New();
    //writer->SetFileName(outArg.getValue());
    //writer->SetImageIO(imageIO);
    //writer->Update();

    return EXIT_SUCCESS;
}
