#pragma once

#include <itkImageToImageMetric.h>
#include <itkCovariantVector.h>
#include <itkPoint.h>
#include <itkNumericTraits.h>

namespace anima
{

template <class TFixedImage, class TMovingImage>
class BaseSymmetricImageToImageMetric : public itk::ImageToImageMetric< TFixedImage, TMovingImage>
{
public:
    /** Standard class typedefs. */
    typedef BaseSymmetricImageToImageMetric Self;
    typedef itk::ImageToImageMetric <TFixedImage, TMovingImage> Superclass;
    typedef itk::SmartPointer <Self> Pointer;
    typedef itk::SmartPointer <const Self> ConstPointer;

    /** Run-time type information (and related methods). */
    itkTypeMacro(BaseSymmetricImageToImageMetric, itk::ImageToImageMetric)

    /** Types transferred from the base class */
    typedef typename Superclass::InterpolatorType InterpolatorType;
    typedef typename Superclass::InterpolatorPointer InterpolatorPointer;
    typedef typename Superclass::TransformType TransformType;
    typedef typename Superclass::TransformPointer TransformPointer;
    typedef typename Superclass::TransformParametersType TransformParametersType;

    typedef typename Superclass::MeasureType MeasureType;
    typedef typename Superclass::DerivativeType DerivativeType;
    typedef typename Superclass::FixedImageType FixedImageType;
    typedef typename FixedImageType::Pointer FixedImagePointer;
    typedef typename Superclass::FixedImageConstPointer FixedImageConstPointer;
    typedef typename Superclass::MovingImageConstPointer MovingImageConstPointer;
    typedef typename itk::NumericTraits <MeasureType>::AccumulateType AccumulateType;
    typedef itk::ContinuousIndex <MeasureType, FixedImageType::ImageDimension> ContinuousIndexType;
    typedef typename FixedImageType::PointType PointType;

    /** Get the derivatives of the match measure. */
    void GetDerivative(const TransformParametersType & parameters,
                       DerivativeType & Derivative) const ITK_OVERRIDE;

    /**  Get the value for single valued optimizers. */
    MeasureType GetValue(const TransformParametersType & parameters) const ITK_OVERRIDE;

    /**  Get value and derivatives for multiple valued optimizers. */
    void GetValueAndDerivative(const TransformParametersType & parameters,
                               MeasureType& Value, DerivativeType& Derivative) const ITK_OVERRIDE;

    itkSetMacro(UseOppositeTransform, bool)
    itkSetMacro(ScaleIntensities, bool)

    itkSetObjectMacro(FixedInterpolator, InterpolatorType)
    itkSetConstObjectMacro(ExtraMiddleImage, FixedImageType)

    itkSetObjectMacro(ReverseTransform, TransformType)

protected:
    BaseSymmetricImageToImageMetric();
    virtual ~BaseSymmetricImageToImageMetric() {}
    virtual void PrintSelf(std::ostream& os, itk::Indent indent) const ITK_OVERRIDE;

    /** Compute specific measure from data */
    virtual MeasureType ComputeMetricFromData(std::vector <AccumulateType> &refData,
                                              std::vector <AccumulateType> &floData) const = 0;

    /** Compute specific measure from data when extra middle image */
    virtual MeasureType ComputeMetricFromFullData(std::vector <AccumulateType> &refData,
                                                  std::vector <AccumulateType> &floData,
                                                  std::vector <AccumulateType> &middleData) const = 0;

private:
    BaseSymmetricImageToImageMetric(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    //We optimize two transforms at a time:
    // Transform allows to go and look with integrated interpolator (moving image)
    // ReverseTransform allows to go and look into fixed image
    //
    mutable TransformPointer m_ReverseTransform;
    mutable InterpolatorPointer m_FixedInterpolator;
    bool m_ScaleIntensities;
    bool m_UseOppositeTransform;

    FixedImageConstPointer m_ExtraMiddleImage;
};

} // end namespace anima

#include "animaBaseSymmetricImageToImageMetric.hxx"
