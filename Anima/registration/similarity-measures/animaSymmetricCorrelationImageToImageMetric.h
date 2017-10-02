#pragma once
#include <animaBaseSymmetricImageToImageMetric.h>

namespace anima
{

template < class TFixedImage, class TMovingImage >
class SymmetricCorrelationImageToImageMetric : public anima::BaseSymmetricImageToImageMetric <TFixedImage, TMovingImage>
{
public:

    /** Standard class typedefs. */
    typedef SymmetricCorrelationImageToImageMetric Self;
    typedef anima::BaseSymmetricImageToImageMetric <TFixedImage, TMovingImage> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SymmetricCorrelationImageToImageMetric, anima::BaseSymmetricImageToImageMetric)

    typedef typename Superclass::RealType RealType;
    typedef typename Superclass::MeasureType MeasureType;
    typedef typename Superclass::AccumulateType AccumulateType;

    itkSetMacro(SquaredCorrelation, bool)

protected:
    SymmetricCorrelationImageToImageMetric();
    virtual ~SymmetricCorrelationImageToImageMetric() {}
    void PrintSelf(std::ostream& os, itk::Indent indent) const ITK_OVERRIDE;

    MeasureType ComputeMetricFromData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData) const ITK_OVERRIDE;
    MeasureType ComputeMetricFromFullData(std::vector <AccumulateType> &refData, std::vector <AccumulateType> &floData,
                                          std::vector <AccumulateType> &middleData) const ITK_OVERRIDE;

private:
    SymmetricCorrelationImageToImageMetric(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    //! Compute squared correlation ? (used only without extra middle image)
    bool m_SquaredCorrelation;
};

} // end namespace anima

#include "animaSymmetricCorrelationImageToImageMetric.hxx"
