#include <animaGaussianMCMCost.h>
#include <cmath>

#include <animaBaseTensorTools.h>

namespace anima
{

GaussianMCMCost::MeasureType
GaussianMCMCost::GetValues(const ParametersType &parameters)
{
    unsigned int numberOfParameters = parameters.GetSize();
    unsigned int nbImages = m_Gradients.size();

    // Set MCM parameters
    m_TestedParameters.resize(numberOfParameters);

    for (unsigned int i = 0;i < numberOfParameters;++i)
        m_TestedParameters[i] = parameters[i];

    m_MCMStructure->SetParametersFromVector(m_TestedParameters);
    
    m_Residuals.SetSize(nbImages);
    m_PredictedSignals.resize(nbImages);
    m_SigmaSquare = 0.0;

    m_LogPriorValue = 0.0;
    if (m_MAPEstimationMode)
        m_LogPriorValue = m_MCMStructure->GetLogPriorValue();

    double priorValue = std::exp(- m_LogPriorValue / nbImages);
    for (unsigned int i = 0;i < nbImages;++i)
    {
        m_PredictedSignals[i] = m_MCMStructure->GetPredictedSignal(m_SmallDelta,m_BigDelta,
                                                                    m_GradientStrengths[i],m_Gradients[i]);

        m_Residuals[i] = m_ObservedSignals[i] - m_PredictedSignals[i];
        m_SigmaSquare += m_Residuals[i] * m_Residuals[i];
        m_Residuals[i] *= priorValue;
    }

    m_SigmaSquare /= nbImages;

    if (m_SigmaSquare < 1.0e-4)
    {
        std::cerr << "Noise variance: " << m_SigmaSquare << std::endl;
        itkExceptionMacro("Too low estimated noise variance.");
    }

    return m_Residuals;
}

double GaussianMCMCost::GetCurrentCostValue()
{
    // This is -2log(L) so that we only have to give one formula
    double costValue = 0;
    unsigned int nbImages = m_Residuals.size();

    if (m_MarginalEstimation)
        costValue = -2.0 * std::log(std::tgamma(1.0 + nbImages / 2.0)) + nbImages * std::log(2.0 * M_PI) + (nbImages + 2.0) * (std::log(nbImages / 2.0) + std::log(m_SigmaSquare) - 2.0 * m_LogPriorValue / nbImages);
    else
        costValue = nbImages * (1.0 + std::log(2.0 * M_PI * m_SigmaSquare)) - 2.0 * m_LogPriorValue;

    return costValue;
}

void
GaussianMCMCost::GetDerivativeMatrix(const ParametersType &parameters, DerivativeMatrixType &derivative)
{
    unsigned int nbParams = parameters.GetSize();
    if (m_MarginalEstimation)
        itkExceptionMacro("Marginal estimation does not boil down to a least square minimization problem.");

    if (nbParams == 0)
        return;

    unsigned int nbValues = m_ObservedSignals.size();

    // Assume get derivative is called with the same parameters as GetValue just before
    for (unsigned int i = 0;i < nbParams;++i)
    {
        if (m_TestedParameters[i] != parameters[i])
            itkExceptionMacro("Get derivative not called with the same parameters as GetValue, suggestive of NaN...");
    }

    derivative.SetSize(nbValues, nbParams);
    derivative.Fill(0.0);

    std::vector<ListType> signalJacobians(nbValues);

    for (unsigned int i = 0;i < nbValues;++i)
    {
        signalJacobians[i] = m_MCMStructure->GetSignalJacobian(m_SmallDelta,m_BigDelta,
                                                               m_GradientStrengths[i],m_Gradients[i]);

        for (unsigned int j = 0;j < nbParams;++j)
            derivative.put(i,j,signalJacobians[i][j]);
    }

    if (m_MAPEstimationMode)
    {
        std::vector <double> priorDerivatives = m_MCMStructure->GetPriorDerivatives();
        double priorValue = std::exp(- m_LogPriorValue / nbValues);
        double diffPriorValue = std::exp(- m_LogPriorValue * (nbValues + 1.0) / nbValues);

        for (unsigned int i = 0;i < nbValues;++i)
        {
            for (unsigned int j = 0;j < nbParams;++j)
            {
                derivative(j,i) *= priorValue;
                derivative(j,i) -= (m_ObservedSignals[i] - m_PredictedSignals[i]) * priorDerivatives[j] * diffPriorValue / nbValues;
            }
        }
    }
}

void
GaussianMCMCost::GetCurrentDerivative(DerivativeMatrixType &derivativeMatrix, DerivativeType &derivative)
{
    unsigned int nbParams = derivativeMatrix.columns();
    unsigned int nbValues = derivativeMatrix.rows();

    derivative.set_size(nbParams);

    // Has to be computed since even in MAP, sigma square is 1/n (y-Falpha)^2
    // We thus miss P^-2/N
    double priorSquared = std::exp(- 2.0 * m_LogPriorValue /nbValues);

    for (unsigned int j = 0;j < nbParams;++j)
    {
        double residualJacobianResidualProduct = 0;
        for (unsigned int i = 0;i < nbValues;++i)
            residualJacobianResidualProduct += derivativeMatrix(i,j) * m_Residuals[i];

        if (!m_MarginalEstimation)
            // Derivative is 2N derivative / sigma^2
            derivative[j] = 2.0 * nbValues * residualJacobianResidualProduct / (m_SigmaSquare * priorSquared);
        else
            derivative[j] = 2.0 * (nbValues + 2.0) * residualJacobianResidualProduct / (nbValues * m_SigmaSquare * priorSquared);
    }
}
    
} // end namespace anima
