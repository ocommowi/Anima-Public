#pragma once

#include <itkVectorImage.h>
#include <animaBaseParticleFilteringTractographyImageFilter.h>
#include <animaODFSphericalHarmonicBasis.h>

#include "AnimaTractographyExport.h"

namespace anima
{

class ANIMATRACTOGRAPHY_EXPORT ODFParticleFilteringTractographyImageFilter : public anima::BaseParticleFilteringTractographyImageFilter < itk::VectorImage <double, 3> >
{
public:
    /** SmartPointer typedef support  */
    typedef ODFParticleFilteringTractographyImageFilter Self;
    typedef BaseParticleFilteringTractographyImageFilter < itk::VectorImage <double, 3> > Superclass;

    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    itkNewMacro(Self)

    itkTypeMacro(ODFParticleFilteringTractographyImageFilter,BaseParticleFilteringTractographyImageFilter)

    void SetODFSHOrder(unsigned int num);
    itkSetMacro(GFAThreshold,double)

protected:
    ODFParticleFilteringTractographyImageFilter();
    virtual ~ODFParticleFilteringTractographyImageFilter();

    //! Generate seed points
    void PrepareTractography() ITK_OVERRIDE;

    virtual Vector3DType ProposeNewDirection(Vector3DType &oldDirection, VectorType &modelValue,
                                             std::mt19937 &random_generator, unsigned int threadId) ITK_OVERRIDE;

    virtual double ComputeLogWeightUpdate(double b0Value, double noiseValue, Vector3DType &previousDirection,
                                          Vector3DType &newDirection, VectorType &previousModelValue, VectorType &modelValue,
                                          unsigned int threadId) ITK_OVERRIDE;

    virtual void ComputeModelValue(InterpolatorPointer &modelInterpolator, ContinuousIndexType &index, VectorType &modelValue) ITK_OVERRIDE;

    virtual bool CheckModelProperties(double estimatedB0Value, double estimatedNoiseValue,
                                      VectorType &modelValue, unsigned int threadId) ITK_OVERRIDE;

    double GetGeneralizedFractionalAnisotropy(VectorType &modelValue);

private:
    ITK_DISALLOW_COPY_AND_ASSIGN(ODFParticleFilteringTractographyImageFilter);

    double m_GFAThreshold;

    unsigned int m_ODFSHOrder;
    anima::ODFSphericalHarmonicBasis *m_ODFSHBasis;
};

} // end of namespace anima