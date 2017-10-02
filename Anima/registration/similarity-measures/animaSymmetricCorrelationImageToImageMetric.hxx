#pragma once
#include "animaSymmetricCorrelationImageToImageMetric.h"

#include <itkImageRegionConstIteratorWithIndex.h>
#include <vnl/algo/vnl_determinant.h>

namespace anima
{

template <class TFixedImage, class TMovingImage>
SymmetricCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::SymmetricCorrelationImageToImageMetric()
{
    m_SquaredCorrelation = true;
}

template <class TFixedImage, class TMovingImage>
typename SymmetricCorrelationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
SymmetricCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeMetricFromData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData) const
{
    AccumulateType squaredMovingSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType fixedMovingSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType movingSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType fixedSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType squaredFixedSum = itk::NumericTraits< AccumulateType >::Zero;

    unsigned int numberOfPixels = refData.size();
    if (numberOfPixels <= 1)
        return 0;

    for (unsigned int i = 0;i < numberOfPixels;++i)
    {
        squaredFixedSum += refData[i] * refData[i];
        squaredMovingSum += floData[i] * floData[i];
        fixedMovingSum += refData[i] * floData[i];
        fixedSum += refData[i];
        movingSum += floData[i];
    }

    RealType fixedVar = squaredFixedSum - fixedSum * fixedSum / numberOfPixels;
    RealType movingVar = squaredMovingSum - movingSum * movingSum / numberOfPixels;
    if ((fixedVar <= 0)||(movingVar <= 0))
        return 0;

    RealType covData = fixedMovingSum - fixedSum * movingSum / numberOfPixels;
    RealType multVars = fixedVar * movingVar;

    MeasureType measure = 0;

    if (m_SquaredCorrelation)
        measure = covData * covData / multVars;
    else
        measure = std::max(0.0,covData / std::sqrt(multVars));

    return measure;
}

template <class TFixedImage, class TMovingImage>
typename SymmetricCorrelationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
SymmetricCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeMetricFromFullData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData,
                            std::vector <AccumulateType> &middleData) const
{
    AccumulateType squaredMovingSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType fixedMovingSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType fixedMiddleSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType movingMiddleSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType movingSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType fixedSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType middleSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType squaredFixedSum = itk::NumericTraits< AccumulateType >::Zero;
    AccumulateType squaredMiddleSum = itk::NumericTraits< AccumulateType >::Zero;

    unsigned int numberOfPixels = refData.size();
    if (numberOfPixels <= 1)
        return 0;

    for (unsigned int i = 0;i < numberOfPixels;++i)
    {
        squaredFixedSum += refData[i] * refData[i];
        squaredMovingSum += floData[i] * floData[i];
        squaredMiddleSum += middleData[i] * middleData[i];
        fixedMovingSum += refData[i] * floData[i];
        fixedMiddleSum += refData[i] * middleData[i];
        movingMiddleSum += middleData[i] * floData[i];
        fixedSum += refData[i];
        movingSum += floData[i];
        middleSum += middleData[i];
    }

    double fixedVar = squaredFixedSum - fixedSum * fixedSum / numberOfPixels;
    double movingVar = squaredMovingSum - movingSum * movingSum / numberOfPixels;
    double middleVar = squaredMiddleSum - middleSum * middleSum / numberOfPixels;
    if ((fixedVar <= 0)||(movingVar <= 0)||(middleVar <= 0))
        return 0;

    double fixedMovingCorr = (fixedMovingSum - fixedSum * movingSum / numberOfPixels) / std::sqrt(fixedVar * movingVar);
    double fixedMiddleCorr = (fixedMiddleSum - fixedSum * middleSum / numberOfPixels) / std::sqrt(fixedVar * middleVar);
    double movingMiddleCorr = (movingMiddleSum - middleSum * movingSum / numberOfPixels) / std::sqrt(middleVar * movingVar);

    if (fixedMovingCorr * fixedMovingCorr >= 1)
        return 1;

    MeasureType resValue = fixedMiddleCorr * fixedMiddleCorr + movingMiddleCorr * movingMiddleCorr;
    resValue -= 2.0 * fixedMovingCorr * fixedMiddleCorr * movingMiddleCorr;
    resValue /= (1.0 - fixedMovingCorr * fixedMovingCorr);

    if (resValue <= 0)
        return 0;

    return std::sqrt(resValue);
}

template <class TFixedImage, class TMovingImage>
void
SymmetricCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
    os << indent << "Squared correlation: " << (m_SquaredCorrelation ? "On" : "Off") << std::endl;
}

} // end namespace anima
