#pragma once
#include "animaSymmetricMeanSquaresImageToImageMetric.h"

#include <itkImageRegionConstIteratorWithIndex.h>
#include <vnl/algo/vnl_determinant.h>

namespace anima
{

template <class TFixedImage, class TMovingImage>
SymmetricMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::SymmetricMeanSquaresImageToImageMetric()
{
}

template <class TFixedImage, class TMovingImage>
typename SymmetricMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
SymmetricMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::ComputeMetricFromData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData) const
{
    MeasureType measure = 0;

    unsigned int numberOfPixels = refData.size();

    if (numberOfPixels == 0)
        return HUGE_VAL;

    for (unsigned int i = 0;i < numberOfPixels;++i)
        measure += (refData[i] - floData[i]) * (refData[i] - floData[i]);

    return measure / numberOfPixels;
}

template <class TFixedImage, class TMovingImage>
typename SymmetricMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
SymmetricMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::ComputeMetricFromFullData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData,
                            std::vector <AccumulateType> &middleData) const
{
    MeasureType measure = 0;

    unsigned int numberOfPixels = refData.size();

    if (numberOfPixels == 0)
        return HUGE_VAL;

    for (unsigned int i = 0;i < numberOfPixels;++i)
    {
        double leftMeasure = (refData[i] - middleData[i]) * (refData[i] - middleData[i]);
        double rightMeasure = (floData[i] - middleData[i]) * (floData[i] - middleData[i]);
        double centerMeasure = (floData[i] - refData[i]) * (floData[i] - refData[i]);
        measure += (leftMeasure + centerMeasure + rightMeasure) / 3.0;
    }

    return measure / numberOfPixels;
}

} // end namespace anima
