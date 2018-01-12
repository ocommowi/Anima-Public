#pragma once
#include "animaPyramidalBlockMatchingBridge.h"

#include <animaReadWriteFunctions.h>
#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>
#include <itkMultiThreader.h>
#include <itkCenteredTransformInitializer.h>

#include <animaAsymmetricBMRegistrationMethod.h>
#include <animaSymmetricBMRegistrationMethod.h>
#include <animaKissingSymmetricBMRegistrationMethod.h>

#include <animaAnatomicalBlockMatcher.h>

#include <animaLSWTransformAgregator.h>
#include <animaLTSWTransformAgregator.h>
#include <animaMEstTransformAgregator.h>

namespace anima
{

template <unsigned int ImageDimension>
PyramidalBlockMatchingBridge<ImageDimension>::PyramidalBlockMatchingBridge()
{
    m_InitialTransform = NULL;
    m_ReferenceImage = NULL;
    m_FloatingImage = NULL;

    m_OutputTransform = NULL;
    m_outputTransformFile = "";

    m_OutputImage = NULL;

    m_BlockSize = 5;
    m_BlockSpacing = 5;
    m_StDevThreshold = 5;

    m_SymmetryType = Asymmetric;
    m_Transform = Translation;
    m_AffineDirection = 1;
    m_Metric = SquaredCorrelation;
    m_Optimizer = Bobyqa;

    m_MaximumIterations = 10;
    m_MinimalTransformError = 0.01;
    m_OptimizerMaximumIterations = 100;
    m_SearchRadius = 2;
    m_SearchAngleRadius = 5;
    m_SearchScaleRadius = 0.1;
    m_FinalRadius = 0.001;
    m_StepSize = 1;
    m_TranslateUpperBound = 50;
    m_AngleUpperBound = 180;
    m_ScaleUpperBound = 3;
    m_Agregator = MEstimation;
    m_OutputTransformType = outRigid;
    m_AgregThreshold = 0.5;
    m_SeStoppingThreshold = 0.01;
    m_NumberOfPyramidLevels = 3;
    m_LastPyramidLevel = 0;
    m_PercentageKept = 0.8;
    m_InitializeOnCenterOfGravity = true;
    m_AdaptBlocksToStructure = false;

    this->SetNumberOfThreads(itk::MultiThreader::GetGlobalDefaultNumberOfThreads());

    m_Abort = false;
    m_Verbose = true;

    m_callback = itk::CStyleCommand::New();
    m_callback->SetClientData ((void *) this);
    m_callback->SetCallback (ManageProgress);
}

template <unsigned int ImageDimension>
PyramidalBlockMatchingBridge<ImageDimension>::~PyramidalBlockMatchingBridge()
{
}

template <unsigned int ImageDimension>
void PyramidalBlockMatchingBridge<ImageDimension>::Abort()
{
    m_Abort = true;

    if(m_bmreg)
        m_bmreg->Abort();
}

template <unsigned int ImageDimension>
void PyramidalBlockMatchingBridge<ImageDimension>::SetInitialTransform (std::string initialTransformFile)
{
    if (initialTransformFile != "")
    {
        itk::TransformFileReader::Pointer tmpTrRead = itk::TransformFileReader::New();
        tmpTrRead->SetFileName (initialTransformFile);

        try
        {
            tmpTrRead->Update();

            itk::TransformFileReader::TransformListType trsfList = * (tmpTrRead->GetTransformList());
            itk::TransformFileReader::TransformListType::iterator tr_it = trsfList.begin();

            m_InitialTransform = dynamic_cast <AffineTransformType *> ((*tr_it).GetPointer());

        }
        catch (itk::ExceptionObject &e)
        {
            std::cerr << "Unable to read initial transform... Exiting..." << std::endl;
        }
    }
}

template <unsigned int ImageDimension>
void PyramidalBlockMatchingBridge<ImageDimension>::Update()
{
    typedef BaseTransformAgregator<ImageDimension> BaseAgreg;

    m_Abort = false;

    // progress management
    m_progressReporter = new itk::ProgressReporter(this, 0, GetNumberOfPyramidLevels()*this->m_MaximumIterations);
    this->AddObserver(itk::ProgressEvent(), m_progressCallback);

    this->InvokeEvent(itk::StartEvent());

    this->SetupPyramids();

    typedef anima::AnatomicalBlockMatcher <InputImageType> BlockMatcherType;

    // Iterate over pyramid levels
    for (unsigned int i = 0;i < GetNumberOfPyramidLevels() && !m_Abort; ++i)
    {
        if (i + GetLastPyramidLevel() >= m_ReferencePyramid->GetNumberOfLevels())
            continue;

        typename InputImageType::Pointer refImage = m_ReferencePyramid->GetOutput(i);
        refImage->DisconnectPipeline();

        typename InputImageType::Pointer floImage = m_FloatingPyramid->GetOutput(i);
        floImage->DisconnectPipeline();

        typename MaskImageType::Pointer maskGenerationImage = 0;
        if (m_BlockGenerationPyramid)
        {
            maskGenerationImage = m_BlockGenerationPyramid->GetOutput(i);
            maskGenerationImage->DisconnectPipeline();
        }

        BlockMatcherType *mainMatcher = new BlockMatcherType;
        BlockMatcherType *reverseMatcher = 0;
        mainMatcher->SetBlockPercentageKept(GetPercentageKept());
        mainMatcher->SetBlockSize(GetBlockSize());
        mainMatcher->SetBlockSpacing(GetBlockSpacing());
        mainMatcher->SetBlockVarianceThreshold(GetStDevThreshold() * GetStDevThreshold());
        mainMatcher->SetBlockGenerationMask(maskGenerationImage);
        mainMatcher->SetAdaptRegionToStructure(m_AdaptBlocksToStructure);

        if (m_Verbose)
        {
            std::cout << "Processing pyramid level " << i << std::endl;
            std::cout << "Image size: " << refImage->GetLargestPossibleRegion().GetSize() << std::endl;
        }

        // Init bm registration method
        switch (m_SymmetryType)
        {
            case Asymmetric:
            {
                typedef typename anima::AsymmetricBMRegistrationMethod <InputImageType> BlockMatchRegistrationType;
                m_bmreg = BlockMatchRegistrationType::New();
                break;
            }

            case Symmetric:
            {
                typedef typename anima::SymmetricBMRegistrationMethod <InputImageType> BlockMatchRegistrationType;
                typename BlockMatchRegistrationType::Pointer tmpReg = BlockMatchRegistrationType::New();

                reverseMatcher = new BlockMatcherType;
                reverseMatcher->SetBlockPercentageKept(GetPercentageKept());
                reverseMatcher->SetBlockSize(GetBlockSize());
                reverseMatcher->SetBlockSpacing(GetBlockSpacing());
                reverseMatcher->SetBlockVarianceThreshold(GetStDevThreshold() * GetStDevThreshold());
                reverseMatcher->SetVerbose(m_Verbose);
                reverseMatcher->SetBlockGenerationMask(maskGenerationImage);
                reverseMatcher->SetAdaptRegionToStructure(m_AdaptBlocksToStructure);

                tmpReg->SetReverseBlockMatcher(reverseMatcher);
                m_bmreg = tmpReg;
                break;
            }

            case Kissing:
            default:
            {
                typedef typename anima::KissingSymmetricBMRegistrationMethod <InputImageType> BlockMatchRegistrationType;
                m_bmreg = BlockMatchRegistrationType::New();
                break;
            }
        }

        mainMatcher->SetVerbose(m_Verbose);
        m_bmreg->SetBlockMatcher(mainMatcher);

        if (m_progressCallback)
        {
            // we cannot connect directly bmreg to m_progressCallback
            // we need to create a new progressReporter with more iterations (m_progressReporter),
            // to listen to progress events from bmreg and to send new ones to m_progressCallback
            m_bmreg->AddObserver(itk::ProgressEvent(), m_callback);
        }

        if (this->GetNumberOfThreads() != 0)
            m_bmreg->SetNumberOfThreads(this->GetNumberOfThreads());

        m_bmreg->SetFixedImage(refImage);
        m_bmreg->SetMovingImage(floImage);

        typedef anima::ResampleImageFilter<InputImageType, InputImageType,
                typename AgregatorType::ScalarType> ResampleFilterType;

        typename ResampleFilterType::Pointer refResampler = ResampleFilterType::New();
        refResampler->SetSize(floImage->GetLargestPossibleRegion().GetSize());
        refResampler->SetOutputOrigin(floImage->GetOrigin());
        refResampler->SetOutputSpacing(floImage->GetSpacing());
        refResampler->SetOutputDirection(floImage->GetDirection());
        refResampler->SetDefaultPixelValue(0);
        refResampler->SetNumberOfThreads(GetNumberOfThreads());
        m_bmreg->SetReferenceImageResampler(refResampler);

        typename ResampleFilterType::Pointer movingResampler = ResampleFilterType::New();
        movingResampler->SetSize(refImage->GetLargestPossibleRegion().GetSize());
        movingResampler->SetOutputOrigin(refImage->GetOrigin());
        movingResampler->SetOutputSpacing(refImage->GetSpacing());
        movingResampler->SetOutputDirection(refImage->GetDirection());
        movingResampler->SetDefaultPixelValue(0);
        movingResampler->SetNumberOfThreads(GetNumberOfThreads());
        m_bmreg->SetMovingImageResampler(movingResampler);

        BaseAgreg *agreg = NULL;

        switch (GetAgregator())
        {
            case LeastSquares:
            {
                typedef LSWTransformAgregator<ImageDimension> Agreg;
                agreg = new Agreg;
                break;
            }

            case LeastTrimmedSquares:
            {
                typedef LTSWTransformAgregator<ImageDimension> Agreg;
                Agreg *tmpAg = new Agreg;

                tmpAg->SetLTSCut(GetAgregThreshold());
                tmpAg->SeStoppingThreshold(GetSeStoppingThreshold());

                agreg = tmpAg;
                break;
            }

            case MEstimation:
            {
                typedef MEstTransformAgregator<ImageDimension> Agreg;
                Agreg *tmpAg = new Agreg;

                tmpAg->SetMEstimateFactor(GetAgregThreshold());
                tmpAg->SeStoppingThreshold(GetSeStoppingThreshold());

                agreg = tmpAg;
                break;
            }
        }

        switch (GetOutputTransformType())
        {
            case outTranslation:
                agreg->SetOutputTransformType(BaseAgreg::TRANSLATION);
                break;
            case outRigid:
                agreg->SetOutputTransformType(BaseAgreg::RIGID);
                break;
            case outAffine:
            default:
                agreg->SetOutputTransformType(BaseAgreg::AFFINE);
                break;
        }

        agreg->SetVerboseAgregation(m_Verbose);
        m_bmreg->SetAgregator(agreg);

        switch (GetTransform())
        {
            case Translation:
                mainMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Translation);
                if (reverseMatcher)
                    reverseMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Translation);
                break;
            case Rigid:
                mainMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Rigid);
                if (reverseMatcher)
                    reverseMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Rigid);
                break;
            case Directional_Affine:
                mainMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Directional_Affine);
                mainMatcher->SetAffineDirection(m_AffineDirection);
                if (reverseMatcher)
                {
                    reverseMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Directional_Affine);
                    reverseMatcher->SetAffineDirection(m_AffineDirection);
                }
                break;
            case Affine:
            default:
                mainMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Affine);
                if (reverseMatcher)
                    reverseMatcher->SetBlockTransformType(BlockMatcherType::Superclass::Affine);
                break;
        }


        switch(GetOptimizer())
        {
            case Exhaustive:
                mainMatcher->SetOptimizerType(BlockMatcherType::Exhaustive);
                if (reverseMatcher)
                    reverseMatcher->SetOptimizerType(BlockMatcherType::Exhaustive);
                break;

            case Bobyqa:
            default:
                mainMatcher->SetOptimizerType(BlockMatcherType::Bobyqa);
                if (reverseMatcher)
                    reverseMatcher->SetOptimizerType(BlockMatcherType::Bobyqa);
                break;
        }

        switch (GetMetric())
        {
            case Correlation:
                mainMatcher->SetSimilarityType(BlockMatcherType::Correlation);
                if (reverseMatcher)
                    reverseMatcher->SetSimilarityType(BlockMatcherType::Correlation);
                break;
            case SquaredCorrelation:
                mainMatcher->SetSimilarityType(BlockMatcherType::SquaredCorrelation);
                if (reverseMatcher)
                    reverseMatcher->SetSimilarityType(BlockMatcherType::SquaredCorrelation);
                break;
            case MeanSquares:
            default:
                mainMatcher->SetSimilarityType(BlockMatcherType::MeanSquares);
                if (reverseMatcher)
                    reverseMatcher->SetSimilarityType(BlockMatcherType::MeanSquares);
                break;
        }

        m_bmreg->SetMaximumIterations(GetMaximumIterations());
        m_bmreg->SetMinimalTransformError(GetMinimalTransformError());
        m_bmreg->SetInitialTransform(m_OutputTransform);

        mainMatcher->SetNumberOfThreads(GetNumberOfThreads());
        mainMatcher->SetOptimizerMaximumIterations(GetOptimizerMaximumIterations());

        double sr = GetSearchRadius();
        mainMatcher->SetSearchRadius(sr);

        double sar = GetSearchAngleRadius();
        mainMatcher->SetSearchAngleRadius(sar);

        double scr = GetSearchScaleRadius();
        mainMatcher->SetSearchScaleRadius(scr);

        double fr = GetFinalRadius();
        mainMatcher->SetFinalRadius(fr);

        double ss = GetStepSize();
        mainMatcher->SetStepSize(ss);

        double tub = GetTranslateUpperBound();
        mainMatcher->SetTranslateMax(tub);

        double aub = GetAngleUpperBound();
        mainMatcher->SetAngleMax(aub);

        double scub = GetScaleUpperBound();
        mainMatcher->SetScaleMax(scub);

        if (reverseMatcher)
        {
            reverseMatcher->SetNumberOfThreads(GetNumberOfThreads());
            reverseMatcher->SetOptimizerMaximumIterations(GetOptimizerMaximumIterations());

            reverseMatcher->SetSearchRadius(sr);
            reverseMatcher->SetSearchAngleRadius(sar);
            reverseMatcher->SetSearchScaleRadius(scr);
            reverseMatcher->SetFinalRadius(fr);
            reverseMatcher->SetStepSize(ss);
            reverseMatcher->SetTranslateMax(tub);
            reverseMatcher->SetAngleMax(aub);
            reverseMatcher->SetScaleMax(scub);
        }

        m_bmreg->SetVerboseProgression(m_Verbose);

        try
        {
            m_bmreg->Update();
        }
        catch( itk::ExceptionObject & err )
        {
            std::cerr << "ExceptionObject caught in bmreg startregistration ! " << err << std::endl;
            exit(-1);
        }

        // Polyrigid will have to be handled here
        AffineTransformType *tmpTrsf = dynamic_cast<AffineTransformType *>(m_OutputTransform.GetPointer());
        tmpTrsf->SetParameters(m_bmreg->GetOutput()->Get()->GetParameters());

        delete mainMatcher;
        if (reverseMatcher)
            delete reverseMatcher;
    }

    if (m_Abort)
        std::cout << "Process aborted" << std::endl;

    this->InvokeEvent(itk::EndEvent());
    this->RemoveAllObservers();

    AffineTransformType *tmpTrsf = dynamic_cast<AffineTransformType *>(m_OutputTransform.GetPointer());
    
    if (m_SymmetryType == Kissing)
    {
        typename AffineTransformType::Pointer trsfCopy = AffineTransformType::New();
        trsfCopy->SetMatrix(tmpTrsf->GetMatrix());
        trsfCopy->SetOffset(tmpTrsf->GetOffset());

        tmpTrsf->Compose(trsfCopy);
    }

    if (!m_InitialTransform.IsNull())
        tmpTrsf->Compose(m_InitialTransform, false);

    typedef typename anima::ResampleImageFilter<InputImageType, InputImageType,
            typename AgregatorType::ScalarType> ResampleFilterType;
    typename ResampleFilterType::Pointer tmpResample = ResampleFilterType::New();
    tmpResample->SetTransform(m_OutputTransform);
    tmpResample->SetInput(m_FloatingImage);

    tmpResample->SetSize(m_ReferenceImage->GetLargestPossibleRegion().GetSize());
    tmpResample->SetOutputOrigin(m_ReferenceImage->GetOrigin());
    tmpResample->SetOutputSpacing(m_ReferenceImage->GetSpacing());
    tmpResample->SetOutputDirection(m_ReferenceImage->GetDirection());
    tmpResample->SetDefaultPixelValue(0);
    tmpResample->Update();

    m_OutputImage = tmpResample->GetOutput();
}

template <unsigned int ImageDimension>
void PyramidalBlockMatchingBridge<ImageDimension>::EmitProgress(int prog)
{
    if (m_progressReporter)
        m_progressReporter->CompletedPixel();
}

template <unsigned int ImageDimension>
void PyramidalBlockMatchingBridge<ImageDimension>::ManageProgress (itk::Object* caller, const itk::EventObject& event, void* clientData)
{
    PyramidalBlockMatchingBridge * source = reinterpret_cast<PyramidalBlockMatchingBridge *> (clientData);
    itk::ProcessObject *processObject = (itk::ProcessObject *) caller;

    if (source && processObject)
        source->EmitProgress(processObject->GetProgress() * 100);
}

template <unsigned int ImageDimension>
void PyramidalBlockMatchingBridge<ImageDimension>::WriteOutputs()
{
    std::cout << "Writing output image to: " << GetResultFile() << std::endl;
    anima::writeImage <InputImageType> (GetResultFile(),m_OutputImage);

    if (GetOutputTransformFile() != "")
    {
        std::cout << "Writing output transform to: " << GetOutputTransformFile() << std::endl;
        itk::TransformFileWriter::Pointer writer = itk::TransformFileWriter::New();
        writer->SetInput(m_OutputTransform);
        writer->SetFileName(GetOutputTransformFile());
        writer->Update();
    }
}

template <unsigned int ImageDimension>
void PyramidalBlockMatchingBridge<ImageDimension>::SetupPyramids()
{
    // Create pyramid here, check images actually are of the same size.
    typedef anima::ResampleImageFilter<InputImageType, InputImageType,
            typename AgregatorType::ScalarType> ResampleFilterType;
    typedef typename itk::CenteredTransformInitializer<AffineTransformType, InputImageType, InputImageType> TransformInitializerType;

    m_ReferencePyramid = PyramidType::New();

    m_ReferencePyramid->SetInput(m_ReferenceImage);
    m_ReferencePyramid->SetNumberOfLevels(GetNumberOfPyramidLevels());
    m_ReferencePyramid->SetNumberOfThreads(GetNumberOfThreads());

    typename ResampleFilterType::Pointer refResampler = ResampleFilterType::New();
    m_ReferencePyramid->SetImageResampler(refResampler);
    m_ReferencePyramid->Update();

    InputImagePointer initialFloatingImage = const_cast <InputImageType *> (m_FloatingImage.GetPointer());

    // Compute initial transform if needed to get a decent initial floating image
    if (!m_InitialTransform.IsNull())
    {
        typename ResampleFilterType::Pointer tmpResample = ResampleFilterType::New();
        tmpResample->SetTransform(m_InitialTransform);
        tmpResample->SetInput(m_FloatingImage);

        tmpResample->SetSize(m_ReferenceImage->GetLargestPossibleRegion().GetSize());
        tmpResample->SetOutputOrigin(m_ReferenceImage->GetOrigin());
        tmpResample->SetOutputSpacing(m_ReferenceImage->GetSpacing());
        tmpResample->SetOutputDirection(m_ReferenceImage->GetDirection());
        tmpResample->SetDefaultPixelValue(0);
        tmpResample->Update();

        initialFloatingImage = tmpResample->GetOutput();
        initialFloatingImage->DisconnectPipeline();
    }
    else
    {
        m_InitialTransform = NULL;

        m_InitialTransform = AffineTransformType::New();
        m_InitialTransform->SetIdentity();

        if (m_InitializeOnCenterOfGravity)
        {
            typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();
            initializer->SetTransform(m_InitialTransform);
            initializer->SetFixedImage(m_ReferenceImage);
            initializer->SetMovingImage(m_FloatingImage);
            initializer->MomentsOn();
            initializer->InitializeTransform();
        }

        typename ResampleFilterType::Pointer tmpResample = ResampleFilterType::New();
        tmpResample->SetTransform(m_InitialTransform);
        tmpResample->SetInput(m_FloatingImage);

        tmpResample->SetSize(m_ReferenceImage->GetLargestPossibleRegion().GetSize());
        tmpResample->SetOutputOrigin(m_ReferenceImage->GetOrigin());
        tmpResample->SetOutputSpacing(m_ReferenceImage->GetSpacing());
        tmpResample->SetOutputDirection(m_ReferenceImage->GetDirection());
        tmpResample->SetDefaultPixelValue(0);
        tmpResample->Update();

        initialFloatingImage = tmpResample->GetOutput();
        initialFloatingImage->DisconnectPipeline();
    }

    // Create pyramid for floating image
    m_FloatingPyramid = PyramidType::New();

    m_FloatingPyramid->SetInput(initialFloatingImage);
    m_FloatingPyramid->SetNumberOfLevels(GetNumberOfPyramidLevels());
    m_FloatingPyramid->SetNumberOfThreads(GetNumberOfThreads());

    typename ResampleFilterType::Pointer floResampler = ResampleFilterType::New();
    m_FloatingPyramid->SetImageResampler(floResampler);

    m_FloatingPyramid->Update();

    m_BlockGenerationPyramid = 0;
    if (m_BlockGenerationMask)
    {
        typedef anima::ResampleImageFilter<MaskImageType, MaskImageType,
                typename AgregatorType::ScalarType> MaskResampleFilterType;

        typename MaskResampleFilterType::Pointer maskResampler = MaskResampleFilterType::New();

        m_BlockGenerationPyramid = MaskPyramidType::New();
        m_BlockGenerationPyramid->SetImageResampler(maskResampler);
        m_BlockGenerationPyramid->SetInput(m_BlockGenerationMask);
        m_BlockGenerationPyramid->SetNumberOfLevels(GetNumberOfPyramidLevels());
        m_BlockGenerationPyramid->SetNumberOfThreads(GetNumberOfThreads());
        m_BlockGenerationPyramid->Update();
    }
}

} // end of namespace anima
