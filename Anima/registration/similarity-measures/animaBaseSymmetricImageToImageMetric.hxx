#pragma once
#include "animaBaseSymmetricImageToImageMetric.h"

#include <itkImageRegionConstIteratorWithIndex.h>
#include <vnl/algo/vnl_determinant.h>

namespace anima
{

template <class TFixedImage, class TMovingImage>
BaseSymmetricImageToImageMetric<TFixedImage,TMovingImage>
::BaseSymmetricImageToImageMetric()
{
    m_UseOppositeTransform = true;
    m_ScaleIntensities = false;
    m_ExtraMiddleImage = 0;
}

template <class TFixedImage, class TMovingImage>
typename BaseSymmetricImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
BaseSymmetricImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
    FixedImageConstPointer fixedImage = this->m_FixedImage;
    if(!fixedImage)
        itkExceptionMacro( << "Fixed image has not been assigned" );

    MovingImageConstPointer movingImage = this->m_Interpolator->GetInputImage();
    if(!movingImage)
        itkExceptionMacro( << "Moving image has not been assigned" );

    if (!m_FixedInterpolator)
        itkExceptionMacro( << "No reverse interpolator set" );

    this->SetTransformParameters(parameters);

    if (m_UseOppositeTransform)
    {
        TransformParametersType oppositeParams = parameters;
        oppositeParams *= -1.0;

        m_ReverseTransform->SetParameters(oppositeParams);
    }
    else
        m_ReverseTransform = this->m_Transform->GetInverseTransform();

    m_FixedInterpolator->SetInputImage(fixedImage);

    unsigned int maxVectorSize = this->GetFixedImageRegion().GetNumberOfPixels();
    std::vector <AccumulateType> refData(maxVectorSize,0), floData(maxVectorSize,0), middleImData(maxVectorSize,0);

    PointType transformedPoint;
    ContinuousIndexType transformedIndex;
    double fixedValue, movingValue;

    typedef typename itk::ImageRegionConstIterator <FixedImageType> FixedIteratorType;
    FixedIteratorType ti(fixedImage, this->GetFixedImageRegion());
    unsigned int numberOfPixels = 0;

    typename FixedImageType::IndexType index;
    PointType inputPoint;
    FixedIteratorType middleImItr;
    if (m_ExtraMiddleImage)
        middleImItr = FixedIteratorType(m_ExtraMiddleImage,this->GetFixedImageRegion());

    while (!ti.IsAtEnd())
    {
        index = ti.GetIndex();
        fixedImage->TransformIndexToPhysicalPoint(index, inputPoint);

        transformedPoint = this->m_Transform->TransformPoint(inputPoint);
        movingImage->TransformPhysicalPointToContinuousIndex(transformedPoint,transformedIndex);

        if (!this->m_Interpolator->IsInsideBuffer(transformedIndex))
        {
            ++ti;
            if (m_ExtraMiddleImage)
                ++middleImItr;

            continue;
        }

        movingValue = this->m_Interpolator->EvaluateAtContinuousIndex(transformedIndex);

        if (m_ScaleIntensities)
        {
            typedef itk::MatrixOffsetTransformBase <typename TransformType::ScalarType,
                    TFixedImage::ImageDimension, TFixedImage::ImageDimension> BaseTransformType;
            BaseTransformType *currentTrsf = dynamic_cast<BaseTransformType *> (this->m_Transform.GetPointer());

            double factor = vnl_determinant(currentTrsf->GetMatrix().GetVnlMatrix());
            movingValue *= factor;
        }

        transformedPoint = m_ReverseTransform->TransformPoint(inputPoint);
        fixedImage->TransformPhysicalPointToContinuousIndex(transformedPoint,transformedIndex);

        if (!m_FixedInterpolator->IsInsideBuffer(transformedIndex))
        {
            ++ti;
            if (m_ExtraMiddleImage)
                ++middleImItr;

            continue;
        }

        fixedValue = m_FixedInterpolator->EvaluateAtContinuousIndex(transformedIndex);

        if (m_ScaleIntensities)
        {
            typedef itk::MatrixOffsetTransformBase <typename TransformType::ScalarType,
                    TFixedImage::ImageDimension, TFixedImage::ImageDimension> BaseTransformType;
            BaseTransformType *currentTrsf = dynamic_cast<BaseTransformType *> (m_ReverseTransform.GetPointer());

            double factor = vnl_determinant(currentTrsf->GetMatrix().GetVnlMatrix());
            fixedValue *= factor;
        }

        refData[numberOfPixels] = fixedValue;
        floData[numberOfPixels] = movingValue;
        if (m_ExtraMiddleImage)
            middleImData[numberOfPixels] = middleImItr.Get();

        ++numberOfPixels;

        if (m_ExtraMiddleImage)
            ++middleImItr;
        ++ti;
    }

    refData.resize(numberOfPixels);
    floData.resize(numberOfPixels);
    if (m_ExtraMiddleImage)
    {
        middleImData.resize(numberOfPixels);
        return this->ComputeMetricFromFullData(refData,floData,middleImData);
    }

    return this->ComputeMetricFromData(refData,floData);
}

template < class TFixedImage, class TMovingImage>
void
BaseSymmetricImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative(const TransformParametersType & parameters,
                DerivativeType & derivative) const
{
    itkExceptionMacro("Derivative not implemented yet...");
}

template <class TFixedImage, class TMovingImage>
void
BaseSymmetricImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType & value, DerivativeType  & derivative) const
{
    itkExceptionMacro("Derivative not implemented yet...");
}

template < class TFixedImage, class TMovingImage>
void
BaseSymmetricImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
    os << indent << "Use opposite transform: " << (m_UseOppositeTransform ? "On" : "Off") << std::endl;
}

} // end namespace anima
