#pragma once
#include <animaBaseSymmetricImageToImageMetric.h>

namespace anima
{

template < class TFixedImage, class TMovingImage >
class SymmetricMeanSquaresImageToImageMetric : public anima::BaseSymmetricImageToImageMetric <TFixedImage, TMovingImage>
{
public:

    /** Standard class typedefs. */
    typedef SymmetricMeanSquaresImageToImageMetric Self;
    typedef anima::BaseSymmetricImageToImageMetric <TFixedImage, TMovingImage> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SymmetricMeanSquaresImageToImageMetric, anima::BaseSymmetricImageToImageMetric)

    typedef typename Superclass::MeasureType MeasureType;
    typedef typename Superclass::AccumulateType AccumulateType;

protected:
    SymmetricMeanSquaresImageToImageMetric();
    virtual ~SymmetricMeanSquaresImageToImageMetric() {}

    MeasureType ComputeMetricFromData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData) const;
    MeasureType ComputeMetricFromFullData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData,
                                          std::vector <AccumulateType> &middleData) const;

private:
    SymmetricMeanSquaresImageToImageMetric(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

} // end namespace anima

#include "animaSymmetricMeanSquaresImageToImageMetric.hxx"
