#include <animaStickCompartment.h>

#include <animaVectorOperations.h>
#include <animaBetaDistribution.h>
#include <animaGammaDistribution.h>
#include <itkSymmetricEigenAnalysis.h>
#include <animaMCMConstants.h>

namespace anima
{

double StickCompartment::GetFourierTransformedDiffusionProfile(double smallDelta, double bigDelta, double gradientStrength, const Vector3DType &gradient)
{
    m_GradientEigenvector1 = gradient[0] * std::sin(this->GetOrientationTheta()) * std::cos(this->GetOrientationPhi())
            + gradient[1] * std::sin(this->GetOrientationTheta()) * std::sin(this->GetOrientationPhi())
            + gradient[2] * std::cos(this->GetOrientationTheta());
    
    double bValue = anima::GetBValueFromAcquisitionParameters(smallDelta, bigDelta, gradientStrength);
    return std::exp(-bValue * (this->GetRadialDiffusivity1() + (this->GetAxialDiffusivity() - this->GetRadialDiffusivity1())
                               * m_GradientEigenvector1 * m_GradientEigenvector1));
}

double StickCompartment::GetLogPriorValue()
{
    double logPriorValue = 0.0;

    if (m_EstimateAxialDiffusivity)
    {
        double faCompartment = this->GetApparentFractionalAnisotropy();
        logPriorValue = anima::GetBetaLogPDF(faCompartment,anima::MCMPriorAlpha,anima::MCMPriorBeta);

        double mdCompartment = this->GetApparentMeanDiffusivity();
        logPriorValue += anima::GetGammaLogPDF(mdCompartment, anima::MCMGammaPriorKDiffusivity, anima::MCMGammaPriorThetaDiffusivity);
    }

    return logPriorValue;
}

StickCompartment::ListType &StickCompartment::GetPriorDerivativeVector()
{
    m_PriorDerivativeVector.resize(this->GetNumberOfParameters());

    // No priors on directions
    m_PriorDerivativeVector[0] = 0.0;
    m_PriorDerivativeVector[1] = 0.0;

    if (m_EstimateAxialDiffusivity)
    {
        // Compute FA derivative
        double radialDiff = this->GetRadialDiffusivity1();
        double diffLambdas = this->GetAxialDiffusivity() - radialDiff;
        double betaPriorDerivative;
        double denomValue = diffLambdas * (diffLambdas + 2.0 * radialDiff) + 3.0 * radialDiff * radialDiff;

        if (denomValue != 0.0)
        {
            betaPriorDerivative = radialDiff * (3.0 * radialDiff + diffLambdas);
            betaPriorDerivative *= std::pow(denomValue,-1.5);
        }
        else
            betaPriorDerivative = 1.0;

        // Multiply by Beta derivative of FA
        double faValue = this->GetApparentFractionalAnisotropy();
        double mdValue = this->GetApparentMeanDiffusivity();
        betaPriorDerivative *= anima::GetBetaPDFDerivative(faValue,anima::MCMPriorAlpha,anima::MCMPriorBeta);

        // Compute individual prior values
        double priorBeta = std::exp(anima::GetBetaLogPDF(faValue,anima::MCMPriorAlpha,anima::MCMPriorBeta));
        double priorLambda = std::exp(anima::GetGammaLogPDF(mdValue,anima::MCMGammaPriorKDiffusivity,anima::MCMGammaPriorThetaDiffusivity));

        // Compute lambda prior derivative
        double lambdaPriorDerivative = anima::GetGammaPDFDerivative(mdValue,anima::MCMGammaPriorKDiffusivity,anima::MCMGammaPriorThetaDiffusivity) / 3.0;

        m_PriorDerivativeVector[2] = betaPriorDerivative * priorLambda + priorBeta * lambdaPriorDerivative;
    }

    return m_PriorDerivativeVector;
}

StickCompartment::ListType &StickCompartment::GetSignalAttenuationJacobian(double smallDelta, double bigDelta, double gradientStrength, const Vector3DType &gradient)
{
    m_JacobianVector.resize(this->GetNumberOfParameters());
    double bValue = anima::GetBValueFromAcquisitionParameters(smallDelta, bigDelta, gradientStrength);
    
    double signalAttenuation = this->GetFourierTransformedDiffusionProfile(smallDelta, bigDelta, gradientStrength, gradient);
    
    // Derivative w.r.t. theta
    m_JacobianVector[0] = -2.0 * bValue * (this->GetAxialDiffusivity() - this->GetRadialDiffusivity1())
            * (gradient[0] * std::cos(this->GetOrientationTheta()) * std::cos(this->GetOrientationPhi())
            + gradient[1] * std::cos(this->GetOrientationTheta()) * std::sin(this->GetOrientationPhi())
            - gradient[2] * std::sin(this->GetOrientationTheta())) * m_GradientEigenvector1 * signalAttenuation;
    
    // Derivative w.r.t. phi
    m_JacobianVector[1] = -2.0 * bValue * std::sin(this->GetOrientationTheta()) * (this->GetAxialDiffusivity() - this->GetRadialDiffusivity1())
            * (gradient[1] * std::cos(this->GetOrientationPhi()) - gradient[0] * std::sin(this->GetOrientationPhi()))
            * m_GradientEigenvector1 * signalAttenuation;
    
    if (m_EstimateAxialDiffusivity)
    {
        // Derivative w.r.t. to d1
        m_JacobianVector[2] = -bValue * m_GradientEigenvector1 * m_GradientEigenvector1 * signalAttenuation;
    }
    
    return m_JacobianVector;
}

double StickCompartment::GetLogDiffusionProfile(const Vector3DType &sample)
{
    Vector3DType compartmentOrientation(0.0);
    anima::TransformSphericalToCartesianCoordinates(this->GetOrientationTheta(),this->GetOrientationPhi(),1.0,compartmentOrientation);

    double scalarProduct = anima::ComputeScalarProduct(compartmentOrientation,sample);

    double resVal = - 1.5 * std::log(2.0 * M_PI) - 0.5 * std::log(this->GetAxialDiffusivity()) - std::log(this->GetRadialDiffusivity1());

    resVal -= (sample.squared_magnitude() - (1.0 - this->GetRadialDiffusivity1() / this->GetAxialDiffusivity()) * scalarProduct * scalarProduct) / (2.0 * this->GetRadialDiffusivity1());

    return resVal;
}

void StickCompartment::SetParametersFromVector(const ListType &params)
{
    if (params.size() != this->GetNumberOfParameters())
        return;

    this->SetOrientationTheta(params[0]);
    this->SetOrientationPhi(params[1]);

    if (m_EstimateAxialDiffusivity)
        this->SetAxialDiffusivity(params[2] + this->GetRadialDiffusivity1());
}

StickCompartment::ListType &StickCompartment::GetParametersAsVector()
{
    m_ParametersVector.resize(this->GetNumberOfParameters());

    m_ParametersVector[0] = this->GetOrientationTheta();
    m_ParametersVector[1] = this->GetOrientationPhi();

    if (m_EstimateAxialDiffusivity)
        m_ParametersVector[2] = this->GetAxialDiffusivity() - this->GetRadialDiffusivity1();

    return m_ParametersVector;
}

StickCompartment::ListType &StickCompartment::GetParameterLowerBounds()
{
    m_ParametersLowerBoundsVector.resize(this->GetNumberOfParameters());
    std::fill(m_ParametersLowerBoundsVector.begin(),m_ParametersLowerBoundsVector.end(),anima::MCMZeroLowerBound);

    return m_ParametersLowerBoundsVector;
}

StickCompartment::ListType &StickCompartment::GetParameterUpperBounds()
{
    m_ParametersUpperBoundsVector.resize(this->GetNumberOfParameters());

    m_ParametersUpperBoundsVector[0] = anima::MCMPolarAngleUpperBound;
    m_ParametersUpperBoundsVector[1] = anima::MCMAzimuthAngleUpperBound;

    if (m_EstimateAxialDiffusivity)
        m_ParametersUpperBoundsVector[2] = anima::MCMDiffusivityUpperBound;

    return m_ParametersUpperBoundsVector;
}

void StickCompartment::SetEstimateAxialDiffusivity(bool arg)
{
    if (m_EstimateAxialDiffusivity == arg)
        return;

    m_EstimateAxialDiffusivity = arg;
    m_ChangedConstraints = true;
}

void StickCompartment::SetCompartmentVector(ModelOutputVectorType &compartmentVector)
{
    if (compartmentVector.GetSize() != this->GetCompartmentSize())
        itkExceptionMacro("The input vector size does not match the size of the compartment");

    Matrix3DType tensor, eVecs;
    vnl_diag_matrix <double> eVals(m_SpaceDimension);
    itk::SymmetricEigenAnalysis <Matrix3DType,vnl_diag_matrix <double>,Matrix3DType> eigSys(m_SpaceDimension);

    unsigned int pos = 0;
    for (unsigned int i = 0;i < m_SpaceDimension;++i)
    {
        for (unsigned int j = 0;j <= i;++j)
        {
            tensor(i,j) = compartmentVector[pos];

            if (i != j)
                tensor(j,i) = compartmentVector[pos];

            ++pos;
        }
    }

    eigSys.ComputeEigenValuesAndVectors(tensor,eVals,eVecs);

    Vector3DType compartmentOrientation, sphDir;

    for (unsigned int i = 0;i < m_SpaceDimension;++i)
        compartmentOrientation[i] = eVecs(2,i);

    anima::TransformCartesianToSphericalCoordinates(compartmentOrientation,sphDir);

    this->SetOrientationTheta(sphDir[0]);
    this->SetOrientationPhi(sphDir[1]);
    this->SetAxialDiffusivity(eVals(2));
    this->SetRadialDiffusivity1((eVals(1) + eVals(0)) / 2.0);
}

unsigned int StickCompartment::GetCompartmentSize()
{
    return 6;
}

unsigned int StickCompartment::GetNumberOfParameters()
{
    if (!m_ChangedConstraints)
        return m_NumberOfParameters;

    // The number of parameters is three: two orientations plus one axial diffusivity
    m_NumberOfParameters = 3;

    if (!m_EstimateAxialDiffusivity)
        --m_NumberOfParameters;

    m_ChangedConstraints = false;
    return m_NumberOfParameters;
}

StickCompartment::ModelOutputVectorType &StickCompartment::GetCompartmentVector()
{
    if (m_CompartmentVector.GetSize() != this->GetCompartmentSize())
        m_CompartmentVector.SetSize(this->GetCompartmentSize());

    Vector3DType compartmentOrientation(0.0);
    anima::TransformSphericalToCartesianCoordinates(this->GetOrientationTheta(),this->GetOrientationPhi(),1.0,compartmentOrientation);

    unsigned int pos = 0;
    for (unsigned int i = 0;i < m_SpaceDimension;++i)
    {
        for (unsigned int j = 0;j <= i;++j)
        {
            m_CompartmentVector[pos] = (this->GetAxialDiffusivity() - this->GetRadialDiffusivity1()) * compartmentOrientation[i] * compartmentOrientation[j];

            if (i == j)
                m_CompartmentVector[pos] += this->GetRadialDiffusivity1();

            ++pos;
        }
    }

    return m_CompartmentVector;
}

const StickCompartment::Matrix3DType &StickCompartment::GetDiffusionTensor()
{
    m_DiffusionTensor.Fill(0);

    for (unsigned int i = 0;i < m_SpaceDimension;++i)
        m_DiffusionTensor(i,i) = this->GetRadialDiffusivity1();

    Vector3DType compartmentOrientation(0.0);
    anima::TransformSphericalToCartesianCoordinates(this->GetOrientationTheta(),this->GetOrientationPhi(),1.0,compartmentOrientation);

    for (unsigned int i = 0;i < m_SpaceDimension;++i)
        for (unsigned int j = i;j < m_SpaceDimension;++j)
        {
            m_DiffusionTensor(i,j) += compartmentOrientation[i] * compartmentOrientation[j] * (this->GetAxialDiffusivity() - this->GetRadialDiffusivity1());
            if (i != j)
                m_DiffusionTensor(j,i) = m_DiffusionTensor(i,j);
        }

    return m_DiffusionTensor;
}

double StickCompartment::GetApparentFractionalAnisotropy()
{
    double l1 = this->GetAxialDiffusivity();
    double l2 = this->GetRadialDiffusivity1();
    double numFA = l1 - l2;
    double denomFA = std::sqrt (l1 * l1 + 2.0 * l2 * l2);

    double fa = 0;
    if (denomFA != 0.0)
        fa = numFA / denomFA;

    return fa;
}

double StickCompartment::GetApparentMeanDiffusivity()
{
    double l1 = this->GetAxialDiffusivity();
    double l2 = this->GetRadialDiffusivity1();

    return (l1 + 2.0 * l2) / 3.0;
}

double StickCompartment::GetApparentParallelDiffusivity()
{
    return this->GetAxialDiffusivity();
}

double StickCompartment::GetApparentPerpendicularDiffusivity()
{
    return this->GetRadialDiffusivity1();
}

} //end namespace anima

