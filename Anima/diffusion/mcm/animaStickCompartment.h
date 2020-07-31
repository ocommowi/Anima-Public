#pragma once

#include <animaBaseCompartment.h>
#include <AnimaMCMExport.h>

namespace anima
{

class ANIMAMCM_EXPORT StickCompartment : public BaseCompartment
{
public:
    // Useful typedefs
    typedef StickCompartment Self;
    typedef BaseCompartment Superclass;
    typedef Superclass::Pointer BasePointer;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;
    typedef Superclass::ModelOutputVectorType ModelOutputVectorType;
    typedef Superclass::Vector3DType Vector3DType;
    typedef Superclass::Matrix3DType Matrix3DType;

    // New macro
    itkNewMacro(Self)

    /** Run-time type information (and related methods) */
    itkTypeMacro(StickCompartment, BaseCompartment)

    DiffusionModelCompartmentType GetCompartmentType() ITK_OVERRIDE {return Stick;}

    virtual double GetFourierTransformedDiffusionProfile(double smallDelta, double bigDelta, double gradientStrength, const Vector3DType &gradient) ITK_OVERRIDE;
    virtual ListType &GetSignalAttenuationJacobian(double smallDelta, double bigDelta, double gradientStrength, const Vector3DType &gradient) ITK_OVERRIDE;
    virtual double GetLogDiffusionProfile(const Vector3DType &sample) ITK_OVERRIDE;

    virtual double GetLogPriorValue() ITK_OVERRIDE;
    virtual ListType &GetPriorDerivativeVector() ITK_OVERRIDE;

    virtual void SetParametersFromVector(const ListType &params) ITK_OVERRIDE;
    virtual ListType &GetParametersAsVector() ITK_OVERRIDE;

    virtual ListType &GetParameterLowerBounds() ITK_OVERRIDE;
    virtual ListType &GetParameterUpperBounds() ITK_OVERRIDE;

    // Set constraints
    void SetEstimateAxialDiffusivity(bool arg);
    void SetCompartmentVector(ModelOutputVectorType &compartmentVector) ITK_OVERRIDE;

    unsigned int GetCompartmentSize() ITK_OVERRIDE;
    unsigned int GetNumberOfParameters() ITK_OVERRIDE;
    ModelOutputVectorType &GetCompartmentVector() ITK_OVERRIDE;

    const Matrix3DType &GetDiffusionTensor() ITK_OVERRIDE;
    double GetApparentFractionalAnisotropy() ITK_OVERRIDE;
    double GetApparentMeanDiffusivity() ITK_OVERRIDE;
    double GetApparentPerpendicularDiffusivity() ITK_OVERRIDE;
    double GetApparentParallelDiffusivity() ITK_OVERRIDE;

protected:
    StickCompartment() : Superclass()
    {
        m_EstimateAxialDiffusivity = true;
        m_ChangedConstraints = true;
        m_GradientEigenvector1 = 0;
    }

    virtual ~StickCompartment() {}

private:
    bool m_EstimateAxialDiffusivity;
    bool m_ChangedConstraints;
    unsigned int m_NumberOfParameters;
    double m_GradientEigenvector1;
};

} //end namespace anima

