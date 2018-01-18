#pragma once
#include "animaFastCorrelationImageToImageMetric.h"

#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkSymmetricEigenAnalysis.h>
#include <vnl_matrix.h>

namespace anima
{

/**
 * Constructor
 */
template <class TFixedImage, class TMovingImage>
FastCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::FastCorrelationImageToImageMetric()
{
    m_SumFixed = 0;
    m_VarFixed = 0;
    m_SquaredCorrelation = true;
    m_ScaleIntensities = false;
    m_AdaptRegionToStructure = false;
    m_FixedImagePoints.clear();
    m_FixedImageValues.clear();
}

/**
 * Get the match Measure
 */
template <class TFixedImage, class TMovingImage>
typename FastCorrelationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
FastCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
    FixedImageConstPointer fixedImage = this->m_FixedImage;

    if( !fixedImage )
    {
        itkExceptionMacro( << "Fixed image has not been assigned" );
    }

    if ( this->m_NumberOfPixelsCounted == 0 )
        return 0;

    MeasureType measure;
    this->SetTransformParameters( parameters );

    typedef typename itk::NumericTraits< MeasureType >::AccumulateType AccumulateType;

    AccumulateType smm = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType sfm = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType sm  = itk::NumericTraits< AccumulateType >::Zero;

    OutputPointType transformedPoint;
    ContinuousIndexType transformedIndex;
    RealType movingValue;

    for (unsigned int i = 0;i < this->m_NumberOfPixelsCounted;++i)
    {
        transformedPoint = this->m_Transform->TransformPoint( m_FixedImagePoints[i] );
        this->m_Interpolator->GetInputImage()->TransformPhysicalPointToContinuousIndex(transformedPoint,transformedIndex);

        if( this->m_Interpolator->IsInsideBuffer( transformedIndex ) )
        {
            movingValue  = this->m_Interpolator->EvaluateAtContinuousIndex( transformedIndex );

            if (m_ScaleIntensities)
            {
                typedef itk::MatrixOffsetTransformBase <typename TransformType::ScalarType,
                                                        TFixedImage::ImageDimension, TFixedImage::ImageDimension> BaseTransformType;
                BaseTransformType *currentTrsf = dynamic_cast<BaseTransformType *> (this->m_Transform.GetPointer());

                double factor = vnl_determinant(currentTrsf->GetMatrix().GetVnlMatrix());
                movingValue *= factor;
            }

            smm += movingValue * movingValue;
            sfm += m_FixedImageValues[i] * movingValue;
            sm += movingValue;
        }
    }

    RealType movingVariance = smm - sm * sm / this->m_NumberOfPixelsCounted;
    if (movingVariance <= 0)
        return 0;

    RealType covData = sfm - m_SumFixed * sm / this->m_NumberOfPixelsCounted;
    RealType multVars = m_VarFixed * movingVariance;

    if (this->m_NumberOfPixelsCounted > 1 && multVars > 0)
    {
        if (m_SquaredCorrelation)
            measure = covData * covData / multVars;
        else
            measure = std::max(0.0,covData / sqrt(multVars));
    }
    else
    {
        measure = itk::NumericTraits< MeasureType >::Zero;
    }

    return measure;
}

template < class TFixedImage, class TMovingImage>
void
FastCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::PreComputeFixedValues()
{
    FixedImageConstPointer fixedImage = this->m_FixedImage;

    if( !fixedImage )
    {
        itkExceptionMacro( << "Fixed image has not been assigned" );
    }

    m_SumFixed = 0;
    m_VarFixed = 0;
    RealType sumSquared = 0;
    RealType fixedValue = 0;

    typedef itk::ImageRegionConstIteratorWithIndex<FixedImageType> FixedIteratorType;

    FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );
    typename FixedImageType::IndexType index;

    this->m_NumberOfPixelsCounted = this->GetFixedImageRegion().GetNumberOfPixels();

    m_FixedImagePoints.resize(this->m_NumberOfPixelsCounted);
    m_FixedImageValues.resize(this->m_NumberOfPixelsCounted);

    InputPointType inputPoint;

    if (!m_AdaptRegionToStructure)
    {
        unsigned int pos = 0;
        while(!ti.IsAtEnd())
        {
            index = ti.GetIndex();
            fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

            m_FixedImagePoints[pos] = inputPoint;
            fixedValue = ti.Value();
            m_FixedImageValues[pos] = fixedValue;

            sumSquared += fixedValue * fixedValue;
            m_SumFixed += fixedValue;

            ++ti;
            ++pos;
        }

        m_VarFixed = sumSquared - m_SumFixed * m_SumFixed / this->m_NumberOfPixelsCounted;
    }
    else
    {
        unsigned int dim = TFixedImage::ImageDimension;
        vnl_matrix <double> structureTensor(dim,dim);
        itk::Vector <double,TFixedImage::ImageDimension> localGradient;
        structureTensor.fill(0.0);

        typename FixedImageType::IndexType tmpIndex;

        while(!ti.IsAtEnd())
        {
            index = ti.GetIndex();
            localGradient.Fill(0.0);

            for (unsigned int i = 0;i < dim;++i)
            {
                tmpIndex = index;
                unsigned int testValue = this->GetFixedImageRegion().GetIndex()[i] + this->GetFixedImageRegion().GetSize()[i] - 1;
                tmpIndex[i] = std::min(testValue,(unsigned int)(index[i] + 1));

                double sizeGap = - tmpIndex[i];
                double dataValueAfter = fixedImage->GetPixel(tmpIndex);
                tmpIndex[i] = std::max(this->GetFixedImageRegion().GetIndex()[i],index[i] - 1);
                double dataValueBefore = fixedImage->GetPixel(tmpIndex);

                sizeGap += tmpIndex[i];
                if (sizeGap != 0.0)
                    localGradient[i] = (dataValueAfter - dataValueBefore) / sizeGap;
            }

            for (unsigned int i = 0;i < dim;++i)
            {
                for (unsigned int j = i;j < dim;++j)
                    structureTensor(i,j) += localGradient[i] * localGradient[j];
            }

            ++ti;
        }

        structureTensor /= this->m_NumberOfPixelsCounted;
        for (unsigned int i = 0;i < dim;++i)
        {
            for (unsigned int j = i + 1;j < dim;++j)
                structureTensor(j,i) = structureTensor(i,j);
        }

        // Now normalize to get right structure tensor
        typedef itk::SymmetricEigenAnalysis < vnl_matrix <double>, vnl_diag_matrix<double>, vnl_matrix <double> > EigenAnalysisType;

        EigenAnalysisType eigen(dim);
        vnl_matrix <double> eigVecs(dim,dim);
        vnl_diag_matrix <double> eigVals(dim);

        eigen.ComputeEigenValuesAndVectors(structureTensor,eigVals,eigVecs);

        if (eigVals[0] != 0)
        {
            double minEig = eigVals[0];
            for (unsigned int i = 0;i < dim;++i)
                eigVals[i] = std::sqrt(minEig / eigVals[i]);
        }
        else
        {
            for (unsigned int i = 0;i < dim;++i)
                eigVals[i] = 1.0;
        }

        double volumeTensor = eigVals[0];
        for (unsigned int i = 1;i < dim;++i)
            volumeTensor *= eigVals[i];

        volumeTensor = std::pow(volumeTensor, 1.0 / dim);

        for (unsigned int i = 0;i < dim;++i)
        {
            double constMinScale = 4.0 / (this->GetFixedImageRegion().GetSize()[i]);
            double constMaxScale = 1.0 / constMinScale;
            eigVals[i] = std::min(constMaxScale,std::max(constMinScale,eigVals[i] / volumeTensor));
        }

        // Now compute transformation of the block (voxel transform)
        vnl_matrix <double> regionTransform(dim,dim);
        std::vector <double> centralPoint(dim);
        std::vector <double> baseTranslation(dim);

        for (unsigned int i = 0;i < dim;++i)
            centralPoint[i] = this->GetFixedImageRegion().GetIndex()[i] + (this->GetFixedImageRegion().GetSize()[i] - 1) / 2.0;

        for (unsigned int i = 0;i < dim;++i)
        {
            baseTranslation[i] = centralPoint[i];
            for (unsigned int j = 0;j < dim;++j)
            {
                regionTransform(i,j) = eigVecs(j,i) * eigVals[j];
                baseTranslation[i] -= regionTransform(i,j) * centralPoint[j];
            }
        }

        ti.GoToBegin();
        unsigned int pos = 0;
        ContinuousIndexType continuousIndex;
        InterpolatorPointer fixedInterpolator = dynamic_cast <InterpolatorType *> (this->m_Interpolator->Clone().GetPointer());
        fixedInterpolator->SetInputImage(fixedImage);

        while(!ti.IsAtEnd())
        {
            index = ti.GetIndex();
            for (unsigned int i = 0;i < dim;++i)
            {
                continuousIndex[i] = baseTranslation[i];
                for (unsigned int j = 0;j < dim;++j)
                    continuousIndex[i] += regionTransform(i,j) * index[j];
            }

            fixedImage->TransformContinuousIndexToPhysicalPoint(continuousIndex, inputPoint);
            m_FixedImagePoints[pos] = inputPoint;

            fixedValue = 0.0;
            if (fixedInterpolator->IsInsideBuffer(continuousIndex))
                fixedValue = fixedInterpolator->EvaluateAtContinuousIndex(continuousIndex);
            m_FixedImageValues[pos] = fixedValue;

            sumSquared += fixedValue * fixedValue;
            m_SumFixed += fixedValue;

            ++ti;
            ++pos;
        }

        m_VarFixed = sumSquared - m_SumFixed * m_SumFixed / this->m_NumberOfPixelsCounted;
    }
}

/**
 * Get the Derivative Measure
 */
template < class TFixedImage, class TMovingImage>
void
FastCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative( const TransformParametersType & parameters,
                DerivativeType & derivative ) const
{
    itkExceptionMacro("Derivative not implemented yet...");
}

/**
 * Get both the match Measure and theDerivative Measure
 */
template <class TFixedImage, class TMovingImage>
void
FastCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType & value, DerivativeType  & derivative) const
{
    itkExceptionMacro("Derivative not implemented yet...");
}

template < class TFixedImage, class TMovingImage>
void
FastCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
    os << indent << m_SumFixed << " " << m_VarFixed << std::endl;
}

} // end of namespace anima
