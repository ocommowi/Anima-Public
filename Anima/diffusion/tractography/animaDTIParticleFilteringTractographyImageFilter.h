#pragma once

#include <itkVectorImage.h>
#include <animaBaseProbabilisticTractographyImageFilter.h>

#include <random>

#include "AnimaTractographyExport.h"

namespace anima
{

class ANIMATRACTOGRAPHY_EXPORT DTIProbabilisticTractographyImageFilter : public BaseProbabilisticTractographyImageFilter < itk::VectorImage <double, 3> >
{
public:
    /** SmartPointer typedef support  */
    typedef DTIProbabilisticTractographyImageFilter Self;
    typedef BaseProbabilisticTractographyImageFilter < itk::VectorImage <double, 3> > Superclass;

    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    itkNewMacro(Self)
    itkTypeMacro(DTIProbabilisticTractographyImageFilter,BaseProbabilisticTractographyImageFilter)

    itkSetMacro(FAThreshold,double)

protected:
    DTIProbabilisticTractographyImageFilter();
    virtual ~DTIProbabilisticTractographyImageFilter();

    virtual Vector3DType ProposeNewDirection(Vector3DType &oldDirection, VectorType &modelValue,
                                             std::mt19937 &random_generator, unsigned int threadId) ITK_OVERRIDE;

    virtual double ComputeLogWeightUpdate(double b0Value, double noiseValue, Vector3DType &previousDirection,
                                          Vector3DType &newDirection, VectorType &previousModelValue, VectorType &modelValue,
                                          unsigned int threadId) ITK_OVERRIDE;

    virtual void ComputeModelValue(InterpolatorPointer &modelInterpolator, ContinuousIndexType &index, VectorType &modelValue) ITK_OVERRIDE;

    virtual bool CheckModelProperties(double estimatedB0Value, double estimatedNoiseValue,
                                      VectorType &modelValue, unsigned int threadId) ITK_OVERRIDE;

    double GetFractionalAnisotropy(VectorType &modelValue);
    void ComputeAdditionalScalarMaps() ITK_OVERRIDE;

private:
    ITK_DISALLOW_COPY_AND_ASSIGN(DTIProbabilisticTractographyImageFilter);

    double m_FAThreshold;
};

} // end of namespace anima
