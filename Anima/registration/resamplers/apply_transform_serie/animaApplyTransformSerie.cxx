#include <tclap/CmdLine.h>

#include <itkResampleImageFilter.h>

#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>
#include <itkImageRegionIterator.h>

#include <itkExtractImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <animaResampleImageFilter.h>
#include <animaTransformSeriesReader.h>
#include <animaReadWriteFunctions.h>
#include <animaRetrieveImageTypeMacros.h>

struct arguments
{
    bool invert;
    unsigned int pthread;
    std::string input, output, geometry, transfo, interpolation;

};

template <class ImageType>
void
applyVectorTransfo(itk::ImageIOBase::Pointer geometryImageIO, const arguments &args)
{
    typedef itk::VectorImage<float, ImageType::ImageDimension> OutputType;

    typedef anima::TransformSeriesReader <double, ImageType::ImageDimension> TransformSeriesReaderType;
    typedef typename TransformSeriesReaderType::OutputTransformType TransformType;
    TransformSeriesReaderType *trReader = new TransformSeriesReaderType;
    trReader->SetInput(args.transfo);
    trReader->SetInvertTransform(args.invert);
    trReader->Update();
    typename TransformType::Pointer transfo = trReader->GetOutputTransform();

    std::cout << "Image to transform is vector." << std::endl;

    typedef itk::ResampleImageFilter<ImageType, OutputType> ResampleFilterType;
    typename ResampleFilterType::Pointer vectorResampler = ResampleFilterType::New();
    typename itk::InterpolateImageFunction <ImageType>::Pointer interpolator;

    if(args.interpolation == "nearest")
        interpolator = itk::NearestNeighborInterpolateImageFunction<ImageType>::New();
    else if(args.interpolation == "linear")
        interpolator = itk::LinearInterpolateImageFunction<ImageType>::New();
    else
    {
        itk::ExceptionObject excp(__FILE__, __LINE__,
                                  "bspline and sinc interpolation not suported for vector images yet.",
                                  ITK_LOCATION);
        throw excp;
    }

    vectorResampler->SetTransform(transfo);
    vectorResampler->SetInterpolator(interpolator);

    typename ImageType::SizeType size;
    typename ImageType::PointType origin;
    typename ImageType::SpacingType spacing;
    typename ImageType::DirectionType direction;
    direction.SetIdentity();
    unsigned int imageIODimension = geometryImageIO->GetNumberOfDimensions();

    for (unsigned int i = 0;i < imageIODimension;++i)
    {
        size[i] = geometryImageIO->GetDimensions(i);
        origin[i] = geometryImageIO->GetOrigin(i);
        spacing[i] = geometryImageIO->GetSpacing(i);
        for(unsigned int j = 0; j < imageIODimension; ++j)
            direction[i][j] = geometryImageIO->GetDirection(j)[i];
    }

    for (unsigned int i = imageIODimension;i < ImageType::ImageDimension;++i)
    {
        size[i] = 1;
        origin[i] = 0;
        spacing[i] = 1;
        direction[i][i] = 1;
    }

    vectorResampler->SetSize(size);
    vectorResampler->SetOutputOrigin(origin);
    vectorResampler->SetOutputSpacing(spacing);
    vectorResampler->SetOutputDirection(direction);

    vectorResampler->SetInput(anima::readImage<ImageType>(args.input));
    vectorResampler->SetNumberOfThreads(args.pthread);
    vectorResampler->Update();

    anima::writeImage<OutputType>(args.output, vectorResampler->GetOutput());
}

template <class ImageType>
void
applyScalarTransfo4D(itk::ImageIOBase::Pointer geometryImageIO, const arguments &args)
{
    typedef itk::Image <float, ImageType::ImageDimension> OutputType;
    const unsigned int InternalImageDimension = 3;
    typedef itk::Image <float, InternalImageDimension> InternalImageType;

    typedef anima::TransformSeriesReader <double, InternalImageDimension> TransformSeriesReaderType;
    typedef typename TransformSeriesReaderType::OutputTransformType TransformType;
    TransformSeriesReaderType *trReader = new TransformSeriesReaderType;
    trReader->SetInput(args.transfo);
    trReader->SetInvertTransform(args.invert);
    trReader->Update();
    typename TransformType::Pointer transfo = trReader->GetOutputTransform();

    std::cout << "Image to transform is 4D scalar." << std::endl;
    typename itk::InterpolateImageFunction <InternalImageType>::Pointer interpolator;

    if(args.interpolation == "nearest")
        interpolator = itk::NearestNeighborInterpolateImageFunction<InternalImageType>::New();
    else if(args.interpolation == "linear")
        interpolator = itk::LinearInterpolateImageFunction<InternalImageType>::New();
    else if(args.interpolation == "bspline")
        interpolator = itk::BSplineInterpolateImageFunction<InternalImageType>::New();
    else if(args.interpolation == "sinc")
    {
        const unsigned int WindowRadius = 4;
        typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
        typedef itk::ConstantBoundaryCondition<InternalImageType> BoundaryConditionType;
        interpolator = itk::WindowedSincInterpolateImageFunction
                <InternalImageType, WindowRadius, WindowFunctionType, BoundaryConditionType, double >::New();
    }

    typename OutputType::PointType origin;
    typename OutputType::SpacingType spacing;
    typename OutputType::DirectionType direction;
    direction.SetIdentity();

    typename OutputType::RegionType outputRegion;
    for (unsigned int i = 0;i < InternalImageDimension;++i)
    {
        outputRegion.SetIndex(i,0);
        outputRegion.SetSize(i,geometryImageIO->GetDimensions(i));
        origin[i] = geometryImageIO->GetOrigin(i);
        spacing[i] = geometryImageIO->GetSpacing(i);
        for(unsigned int j = 0;j < InternalImageDimension;++j)
            direction(i,j) = geometryImageIO->GetDirection(j)[i];
    }

    typename ImageType::Pointer inputImage = anima::readImage<ImageType> (args.input);
    for (unsigned int i = InternalImageDimension;i < ImageType::ImageDimension;++i)
    {
        outputRegion.SetIndex(i,0);
        outputRegion.SetSize(i,inputImage->GetLargestPossibleRegion().GetSize()[i]);
        origin[i] = inputImage->GetOrigin()[i];
        spacing[i] = inputImage->GetSpacing()[i];
        direction(i,i) = inputImage->GetDirection()(i,i);
    }

    typename OutputType::Pointer outputImage = OutputType::New();
    outputImage->Initialize();
    outputImage->SetRegions(outputRegion);
    outputImage->SetOrigin(origin);
    outputImage->SetSpacing(spacing);
    outputImage->SetDirection(direction);
    outputImage->Allocate();

    unsigned int numImages = inputImage->GetLargestPossibleRegion().GetSize()[InternalImageDimension];

    for (unsigned int i = 0;i < numImages;++i)
    {
        std::cout << "Resampling sub-image " << i+1 << "/" << numImages << std::endl;
        typedef anima::ResampleImageFilter<InternalImageType, InternalImageType> ResampleFilterType;
        typename ResampleFilterType::Pointer scalarResampler = ResampleFilterType::New();
        scalarResampler->SetTransform(transfo);
        scalarResampler->SetInterpolator(interpolator);

        typename InternalImageType::SizeType internalSize;
        typename InternalImageType::PointType internalOrigin;
        typename InternalImageType::SpacingType internalSpacing;
        typename InternalImageType::DirectionType internalDirection;
        internalDirection.SetIdentity();

        for (unsigned int j = 0;j < InternalImageDimension;++j)
        {
            internalSize[j] = geometryImageIO->GetDimensions(j);
            internalOrigin[j] = geometryImageIO->GetOrigin(j);
            internalSpacing[j] = geometryImageIO->GetSpacing(j);
            for(unsigned int k = 0;k < InternalImageDimension;++k)
                internalDirection(j,k) = geometryImageIO->GetDirection(k)[j];
        }

        scalarResampler->SetSize(internalSize);
        scalarResampler->SetOutputOrigin(internalOrigin);
        scalarResampler->SetOutputSpacing(internalSpacing);
        scalarResampler->SetOutputDirection(internalDirection);

        typedef itk::ExtractImageFilter <ImageType, InternalImageType> ExtractFilterType;
        typename ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
        extractFilter->SetInput(inputImage);
        extractFilter->SetDirectionCollapseToGuess();

        typename ImageType::RegionType extractRegion = inputImage->GetLargestPossibleRegion();
        extractRegion.SetIndex(InternalImageDimension,i);
        extractRegion.SetSize(InternalImageDimension,0);

        extractFilter->SetExtractionRegion(extractRegion);
        extractFilter->SetNumberOfThreads(args.pthread);

        extractFilter->Update();

        scalarResampler->SetInput(extractFilter->GetOutput());
        scalarResampler->SetNumberOfThreads(args.pthread);
        scalarResampler->Update();

        extractRegion.SetSize(InternalImageDimension,1);
        for (unsigned int j = 0;j < InternalImageDimension;++j)
        {
            extractRegion.SetIndex(j,scalarResampler->GetOutput()->GetLargestPossibleRegion().GetIndex()[j]);
            extractRegion.SetSize(j,scalarResampler->GetOutput()->GetLargestPossibleRegion().GetSize()[j]);
        }

        itk::ImageRegionIterator <InternalImageType> internalOutItr(scalarResampler->GetOutput(),
                                                                    scalarResampler->GetOutput()->GetLargestPossibleRegion());
        itk::ImageRegionIterator <OutputType> outItr(outputImage,extractRegion);

        while (!outItr.IsAtEnd())
        {
            outItr.Set(internalOutItr.Get());

            ++internalOutItr;
            ++outItr;
        }
    }

    anima::writeImage<OutputType>(args.output, outputImage);
}

template <class ImageType>
void
applyScalarTransfo(itk::ImageIOBase::Pointer geometryImageIO, const arguments &args)
{
    typedef itk::Image <float, ImageType::ImageDimension> OutputType;

    typedef anima::TransformSeriesReader <double, ImageType::ImageDimension> TransformSeriesReaderType;
    typedef typename TransformSeriesReaderType::OutputTransformType TransformType;
    TransformSeriesReaderType *trReader = new TransformSeriesReaderType;
    trReader->SetInput(args.transfo);
    trReader->SetInvertTransform(args.invert);
    trReader->Update();
    typename TransformType::Pointer transfo = trReader->GetOutputTransform();

    std::cout << "Image to transform is scalar" << std::endl;
    typedef anima::ResampleImageFilter<ImageType, OutputType> ResampleFilterType;
    typename ResampleFilterType::Pointer scalarResampler = ResampleFilterType::New();
    typename itk::InterpolateImageFunction <ImageType>::Pointer interpolator;

    if(args.interpolation == "nearest")
        interpolator = itk::NearestNeighborInterpolateImageFunction<ImageType>::New();
    else if(args.interpolation == "linear")
        interpolator = itk::LinearInterpolateImageFunction<ImageType>::New();
    else if(args.interpolation == "bspline")
        interpolator = itk::BSplineInterpolateImageFunction<ImageType>::New();
    else if(args.interpolation == "sinc")
    {
        const unsigned int WindowRadius = 4;
        typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
        typedef itk::ConstantBoundaryCondition<ImageType> BoundaryConditionType;
        interpolator = itk::WindowedSincInterpolateImageFunction
                <ImageType, WindowRadius, WindowFunctionType, BoundaryConditionType, double >::New();
    }

    scalarResampler->SetTransform(transfo);
    scalarResampler->SetInterpolator(interpolator);

    typename ImageType::SizeType size;
    typename ImageType::PointType origin;
    typename ImageType::SpacingType spacing;
    typename ImageType::DirectionType direction;
    direction.SetIdentity();
    unsigned int imageIODimension = geometryImageIO->GetNumberOfDimensions();

    for (unsigned int i = 0;i < imageIODimension;++i)
    {
        size[i] = geometryImageIO->GetDimensions(i);
        origin[i] = geometryImageIO->GetOrigin(i);
        spacing[i] = geometryImageIO->GetSpacing(i);
        for(unsigned int j = 0; j < imageIODimension; ++j)
            direction[i][j] = geometryImageIO->GetDirection(j)[i];
    }

    scalarResampler->SetSize(size);
    scalarResampler->SetOutputOrigin(origin);
    scalarResampler->SetOutputSpacing(spacing);
    scalarResampler->SetOutputDirection(direction);

    scalarResampler->SetInput(anima::readImage<ImageType>(args.input));
    scalarResampler->SetNumberOfThreads(args.pthread);
    scalarResampler->Update();

    anima::writeImage<OutputType>(args.output, scalarResampler->GetOutput());
}

template <class ComponentType, int Dimension>
void
checkIfComponentsAreVectors(itk::ImageIOBase::Pointer inputImageIO, itk::ImageIOBase::Pointer geometryImageIO, const arguments &args)
{
    if (inputImageIO->GetNumberOfComponents() > 1)
    {
        if (Dimension > 3)
            throw itk::ExceptionObject (__FILE__, __LINE__, "Number of dimensions not supported for vector image resampling", ITK_LOCATION);

        applyVectorTransfo < itk::VectorImage<ComponentType, 3> > (geometryImageIO, args);
    }
    else
    {
        if (Dimension < 4)
            applyScalarTransfo < itk::Image<ComponentType, 3> > (geometryImageIO, args);
        else
            applyScalarTransfo4D < itk::Image<ComponentType, 4> > (geometryImageIO, args);
    }
}

template <class ComponentType>
void
retrieveNbDimensions(itk::ImageIOBase::Pointer inputImageIO, itk::ImageIOBase::Pointer geometryImageIO,  const arguments &args)
{
    if (inputImageIO->GetNumberOfDimensions() > 4)
        throw itk::ExceptionObject(__FILE__, __LINE__, "Number of dimensions not supported.", ITK_LOCATION);

    if (inputImageIO->GetNumberOfDimensions() > 3)
        checkIfComponentsAreVectors<ComponentType, 4> (inputImageIO, geometryImageIO, args);
    else
        checkIfComponentsAreVectors<ComponentType, 3> (inputImageIO, geometryImageIO, args);
}


int main(int ac, const char** av)
{
    std::string descriptionMessage = "Resampler tool to apply a series of transformations to an image.\n"
                                     "Input transform is an XML file describing all transforms to apply.\n"
                                     "Such args file should look like this:\n"
                                     "<TransformationList>\n"
                                     "<Transformation>\n"
                                     "<Type>linear</Type> (it can be svf or dense too)\n"
                                     "<Path>FileName</Path>\n"
                                     "<Inversion>0</Inversion>\n"
                                     "</Transformation>\n"
                                     "...\n"
                                     "</TransformationList>\n"
                                     "Note that only geometries and input with the same number of dimensions are supported for now.\n"
                                     "The default interpolation method is linear, it can be [nearest, linear, bspline, sinc]."
                                     "INRIA / IRISA - VisAGeS Team";

    TCLAP::CmdLine cmd(descriptionMessage, ' ',ANIMA_VERSION);

    TCLAP::ValueArg<std::string> inArg("i","input","Input image",true,"","input image",cmd);
    TCLAP::ValueArg<std::string> trArg("t","trsf","Transformations XML list",true,"","transformations list",cmd);
    TCLAP::ValueArg<std::string> outArg("o","output","Output resampled image",true,"","output image",cmd);
    TCLAP::ValueArg<std::string> geomArg("g","geometry","Geometry image",true,"","geometry image",cmd);

    TCLAP::SwitchArg invertArg("I","invert","Invert the transformation series",cmd,false);
    TCLAP::ValueArg<std::string> interpolationArg("n",
                                                  "interpolation",
                                                  "interpolation method to use [nearest, linear, bspline, sinc]",
                                                  false,
                                                  "linear",
                                                  "interpolation method",
                                                  cmd);

    TCLAP::ValueArg<unsigned int> nbpArg("p","numberofthreads","Number of threads to run on (default : all cores)",
                                         false,itk::MultiThreader::GetGlobalDefaultNumberOfThreads(),"number of threads",cmd);

    try
    {
        cmd.parse(ac,av);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    // Find out the type of the image in file
    itk::ImageIOBase::Pointer inputImageIO = itk::ImageIOFactory::CreateImageIO(inArg.getValue().c_str(),
                                                                           itk::ImageIOFactory::ReadMode);
    itk::ImageIOBase::Pointer geometryImageIO = itk::ImageIOFactory::CreateImageIO(geomArg.getValue().c_str(),
                                                                           itk::ImageIOFactory::ReadMode);
    if(!inputImageIO)
    {
        std::cerr << "Itk could not find suitable IO factory for the input." << std::endl;
        return EXIT_FAILURE;
    }
    if(!geometryImageIO)
    {
        std::cerr << "Itk could not find suitable IO factory for the geometry image." << std::endl;
        return EXIT_FAILURE;
    }
    if(inputImageIO->GetNumberOfDimensions() != geometryImageIO->GetNumberOfDimensions())
    {
        std::cerr << "Input and geometry images have to have the same number of dimensions." << std::endl;
        return EXIT_FAILURE;
    }

    // Now that we found the appropriate ImageIO class, ask it to read the meta data from the image file.
    inputImageIO->SetFileName(inArg.getValue());
    inputImageIO->ReadImageInformation();
    geometryImageIO->SetFileName(geomArg.getValue());
    geometryImageIO->ReadImageInformation();

    arguments args;
    args.input = inArg.getValue();
    args.output = outArg.getValue();
    args.geometry = geomArg.getValue();
    args.transfo = trArg.getValue();
    args.invert = invertArg.getValue();
    args.pthread = nbpArg.getValue();
    args.interpolation = interpolationArg.getValue();

    bool badInterpolation = true;
    std::string interpolations[4] = {"nearest", "linear", "bspline", "sinc"};
    for(int i = 0; i < 4; ++i)
    {
        if(args.interpolation == interpolations[i])
        {
            badInterpolation = false;
            break;
        }
    }

    if(badInterpolation)
         std::cerr << "Interpolation method not suported, it must be one of [nearest, linear, bspline, sinc]." << std::endl;

    try
    {
        ANIMA_RETRIEVE_COMPONENT_TYPE(inputImageIO,
                                      retrieveNbDimensions,
                                      inputImageIO,
                                      geometryImageIO,
                                      args)
    }
    catch ( itk::ExceptionObject & err )
    {
        std::cerr << "Can't apply transformation, be sure to use valid arguments..." << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
