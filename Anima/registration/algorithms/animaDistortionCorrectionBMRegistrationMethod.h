#pragma once
#include <animaBaseBMRegistrationMethod.h>

namespace anima
{

template <typename TInputImageType>
class DistortionCorrectionBMRegistrationMethod : public anima::BaseBMRegistrationMethod <TInputImageType>
{
public:
    /** Standard class typedefs. */
    typedef DistortionCorrectionBMRegistrationMethod Self;
    typedef BaseBMRegistrationMethod <TInputImageType> Superclass;
    typedef itk::SmartPointer <Self> Pointer;
    typedef itk::SmartPointer <const Self> ConstPointer;

    enum AttractionModeDefinition
    {
        Symmetric = 0,
        Untargeted_Attraction,
        Targeted_Attraction
    };

    typedef typename Superclass::ImageScalarType ImageScalarType;
    typedef typename Superclass::InputImageType InputImageType;
    typedef typename Superclass::InputImagePointer InputImagePointer;
    typedef typename Superclass::TransformType TransformType;
    typedef typename Superclass::TransformPointer TransformPointer;
    typedef typename Superclass::BlockMatcherType BlockMatcherType;
    typedef typename Superclass::AgregatorType AgregatorType;
    typedef typename Superclass::AgregatorScalarType AgregatorScalarType;
    typedef typename Superclass::AffineTransformType AffineTransformType;
    typedef typename Superclass::SVFTransformType SVFTransformType;
    typedef typename Superclass::DisplacementFieldTransformType DisplacementFieldTransformType;
    typedef typename Superclass::DisplacementFieldTransformPointer DisplacementFieldTransformPointer;

    /** Run-time type information (and related methods). */
    itkTypeMacro(DistortionCorrectionBMRegistrationMethod, BaseBMRegistrationMethod)

    itkNewMacro(Self)

    itkSetMacro(CurrentTransform, TransformPointer)
    itkSetMacro(AttractionMode, AttractionModeDefinition)

protected:
    DistortionCorrectionBMRegistrationMethod() {}
    virtual ~DistortionCorrectionBMRegistrationMethod() {}

    virtual void SetupTransform(TransformPointer &optimizedTransform) ITK_OVERRIDE;
    virtual void ResampleImages(TransformType *currentTransform, InputImagePointer &refImage, InputImagePointer &movingImage) ITK_OVERRIDE;
    virtual bool ComposeAddOnWithTransform(TransformPointer &computedTransform, TransformType *addOn) ITK_OVERRIDE;

    virtual void PerformOneIteration(InputImageType *refImage, InputImageType *movingImage, TransformPointer &addOn) ITK_OVERRIDE;
    void PerformOneIterationAttractor(InputImageType *refImage, InputImageType *movingImage, TransformPointer &addOn);
    void PerformOneIterationSymmetric(InputImageType *refImage, InputImageType *movingImage, TransformPointer &addOn);

private:
    DistortionCorrectionBMRegistrationMethod(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    TransformPointer m_CurrentTransform;
    AttractionModeDefinition m_AttractionMode;
};

} // end namespace anima

#include "animaDistortionCorrectionBMRegistrationMethod.hxx"
