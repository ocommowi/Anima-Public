#pragma once

#include <animaBaseParticleFilteringTractographyImageFilter.h>
#include <animaMultiCompartmentModel.h>
#include <animaMCMImage.h>

#include "AnimaTractographyExport.h"

namespace anima
{

class ANIMATRACTOGRAPHY_EXPORT MCMParticleFilteringTractographyImageFilter : public BaseParticleFilteringTractographyImageFilter < anima::MCMImage <double, 3> >
{
public:
    /** SmartPointer typedef support  */
    typedef MCMParticleFilteringTractographyImageFilter Self;
    typedef BaseParticleFilteringTractographyImageFilter < anima::MCMImage <double, 3> > Superclass;

    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    itkNewMacro(Self)
    itkTypeMacro(MCMParticleFilteringTractographyImageFilter,BaseParticleFilteringTractographyImageFilter)

    typedef anima::MultiCompartmentModel MCModelType;
    typedef MCModelType::Pointer MCModelPointer;

    void SetInputModelImage(InputModelImageType *image) ITK_OVERRIDE
    {
        this->Superclass::SetInputModelImage(image);
        this->SetModelDimension(image->GetDescriptionModel()->GetSize());

        m_WorkModels.resize(itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads());
        for (unsigned int i = 0;i < m_WorkModels.size();++i)
            m_WorkModels[i] = image->GetDescriptionModel()->Clone();
    }

    itkSetMacro(FAThreshold,double)
    itkSetMacro(IsotropicThreshold,double)

    InterpolatorType *GetModelInterpolator() ITK_OVERRIDE;

protected:
    MCMParticleFilteringTractographyImageFilter();
    virtual ~MCMParticleFilteringTractographyImageFilter();

    virtual Vector3DType ProposeNewDirection(Vector3DType &oldDirection, VectorType &modelValue,
                                             std::mt19937 &random_generator, unsigned int threadId) ITK_OVERRIDE;

    virtual double ComputeLogWeightUpdate(double b0Value, double noiseValue, Vector3DType &previousDirection,
                                          Vector3DType &newDirection, VectorType &previousModelValue, VectorType &modelValue,
                                          unsigned int threadId) ITK_OVERRIDE;

    virtual void ComputeModelValue(InterpolatorPointer &modelInterpolator, ContinuousIndexType &index, VectorType &modelValue) ITK_OVERRIDE;

    virtual bool CheckModelProperties(double estimatedB0Value, double estimatedNoiseValue, VectorType &modelValue,
                                      unsigned int threadId) ITK_OVERRIDE;

private:
    ITK_DISALLOW_COPY_AND_ASSIGN(MCMParticleFilteringTractographyImageFilter);

    std::vector <MCModelPointer> m_WorkModels;

    double m_FAThreshold;
    double m_IsotropicThreshold;
};

} // end of namespace anima
