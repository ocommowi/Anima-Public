#include "animaMCMParticleFilteringTractographyImageFilter.h"

#include <animaVectorOperations.h>
#include <animaMatrixOperations.h>
#include <animaLogarithmFunctions.h>
#include <animaDistributionSampling.h>
#include <animaVMFDistribution.h>
#include <animaWatsonDistribution.h>
#include <animaBaseCompartment.h>
#include <animaMCMConstants.h>
#include <animaMCMLinearInterpolateImageFunction.h>

namespace anima
{

MCMParticleFilteringTractographyImageFilter::MCMParticleFilteringTractographyImageFilter(): BaseParticleFilteringTractographyImageFilter()
{
    m_FAThreshold = 0.5;

    // Useless here, defined on the fly as MCM image is set
    this->SetModelDimension(1);
    m_IsotropicThreshold = 0.8;
}

MCMParticleFilteringTractographyImageFilter::~MCMParticleFilteringTractographyImageFilter()
{
}

MCMParticleFilteringTractographyImageFilter::InterpolatorType *
MCMParticleFilteringTractographyImageFilter::GetModelInterpolator()
{
    typedef anima::MCMLinearInterpolateImageFunction <InputModelImageType> MCMInterpolatorType;
    typedef MCMInterpolatorType::Pointer MCMInterpolatorPointer;

    MCMInterpolatorPointer internalInterpolator = MCMInterpolatorType::New();
    internalInterpolator->SetInputImage(this->GetInputModelImage());

    MCModelPointer tmpMCM = this->GetInputModelImage()->GetDescriptionModel()->Clone();
    internalInterpolator->SetReferenceOutputModel(tmpMCM);

    internalInterpolator->Register();
    return internalInterpolator;
}

MCMParticleFilteringTractographyImageFilter::Vector3DType
MCMParticleFilteringTractographyImageFilter::ProposeNewDirection(Vector3DType &oldDirection, VectorType &modelValue,
                                                                 std::mt19937 &random_generator, unsigned int threadId)
{
    bool is2d = (this->GetInputModelImage()->GetLargestPossibleRegion().GetSize()[2] == 1);

    m_WorkModels[threadId]->SetModelVector(modelValue);
    MCModelType::Vector3DType direction;
    m_WorkModels[threadId]->GetRandomlySampledDirection(random_generator, false, direction);

    Vector3DType resVec;
    for (unsigned int i = 0;i < 3;++i)
        resVec[i] = direction[i];

    if (is2d)
    {
        resVec[InputModelImageType::ImageDimension - 1] = 0;
        resVec.Normalize();
    }

    if (anima::ComputeScalarProduct(oldDirection, resVec) < 0)
        resVec *= -1;
    
    return resVec;
}

bool MCMParticleFilteringTractographyImageFilter::CheckModelProperties(double estimatedB0Value, double estimatedNoiseValue, VectorType &modelValue,
                                                                       unsigned int threadId)
{
    // Prevent fibers from going outside of brain mask
    if (estimatedB0Value < 10.0)
        return false;

    // SNR too high means not in white matter anymore
    double snr = estimatedB0Value / std::sqrt(estimatedNoiseValue);
    if (snr > 60.0)
        return false;

    // if all fixels are damaged, stop extending fibers
    unsigned int numIsoCompartments = this->GetInputModelImage()->GetDescriptionModel()->GetNumberOfIsotropicCompartments();
    unsigned int numberOfFixels = this->GetInputModelImage()->GetDescriptionModel()->GetNumberOfCompartments();
    bool allFixelsDamaged = true;

    m_WorkModels[threadId]->SetModelVector(modelValue);
    for (unsigned int i = numIsoCompartments;i < numberOfFixels;++i)
    {
        if (m_WorkModels[threadId]->GetCompartment(i)->GetExtraAxonalFraction() < m_IsotropicThreshold)
        {
            allFixelsDamaged = false;
            break;
        }
    }
    
    if (allFixelsDamaged)
        return false;
    
    // if free water is too important, stop fibers
    double isotropicProportion = 0;
    for (unsigned int i = 0;i < numIsoCompartments;++i)
        isotropicProportion += m_WorkModels[threadId]->GetCompartmentWeight(i);

    if (isotropicProportion > m_IsotropicThreshold)
        return false;

    return true;
}

double MCMParticleFilteringTractographyImageFilter::ComputeLogWeightUpdate(double b0Value, double noiseValue, Vector3DType &previousDirection,
                                                                           Vector3DType &newDirection, VectorType &previousModelValue, VectorType &modelValue,
                                                                           unsigned int threadId)
{
    // Prior is a Watson PDF
    double log_prior = anima::safe_log(anima::EvaluateWatsonPDF(newDirection, previousDirection, this->GetKappaOfPriorDistribution()));

    // Proposal is the probability to get the sampled new direction from the previous model. We assume the ODF integral is 1
    m_WorkModels[threadId]->SetModelVector(previousModelValue);
    MCModelType::Vector3DType tmpDirection;
    for (unsigned int i = 0;i < 3;++i)
        tmpDirection[i] = newDirection[i];
    double log_proposal = std::log(m_WorkModels[threadId]->GetAlongDirectionDiffusionProfileIntegral(tmpDirection,false));

    // Log-likelihood: simulate from model at arrival and from current direction viewed as generic zeppelin tensor
    double log_likelihood = 0;
    m_WorkModels[threadId]->SetModelVector(modelValue);

    unsigned int numGradientImages = this->GetNumberOfGradientDirections();
    MCModelType::Vector3DType gradient;
    vnl_matrix <double> zeppelinDirection(3,3);
    zeppelinDirection.fill(0.0);

    for (unsigned int i = 0;i < 3;++i)
    {
        zeppelinDirection(i,i) = 1.5e-4 + (1.71e-3 - 1.5e-4) * newDirection[i] * newDirection[i];
        for (unsigned int j = i + 1;j < 3;++j)
        {
            zeppelinDirection(i,j) = (1.71e-3 - 1.5e-4) * newDirection[i] * newDirection[j];
            zeppelinDirection(j,i) = zeppelinDirection(i,j);
        }
    }

    for (unsigned int i = 0;i < numGradientImages;++i)
    {
        double internalProductDirection = 0.0;
        double bvalue = this->GetBValueItem(i);
        double gradientStrength = anima::GetGradientStrengthFromBValue(bvalue,anima::DiffusionSmallDelta, anima::DiffusionBigDelta);
        for (unsigned int j = 0;j < 3;++j)
            gradient[j] = this->GetDiffusionGradient(i)[j];
        double predictedSignalMCM = m_WorkModels[threadId]->GetPredictedSignal(anima::DiffusionSmallDelta, anima::DiffusionBigDelta,
                                                                               gradientStrength, gradient);

        for (unsigned int j = 0;j < 3;++j)
        {
            internalProductDirection += gradient[j] * gradient[j] * zeppelinDirection(j,j);

            for (unsigned int k = j + 1;k < 3;++k)
                internalProductDirection += 2.0 * gradient[j] * gradient[k] * zeppelinDirection(j,k);
        }

        internalProductDirection *= bvalue;

        double diffValue = b0Value * (predictedSignalMCM - std::exp(- internalProductDirection));
        double logNoiseGaussValue = 0.5 * (std::log(noiseValue) + std::log(2.0 * M_PI) + diffValue * diffValue / noiseValue);

        log_likelihood -= logNoiseGaussValue;
    }

    double resVal = log_prior + log_likelihood - log_proposal;
    return resVal;
}

void MCMParticleFilteringTractographyImageFilter::ComputeModelValue(InterpolatorPointer &modelInterpolator, ContinuousIndexType &index, VectorType &modelValue)
{
    modelValue.SetSize(this->GetModelDimension());
    modelValue.Fill(0.0);

    if (modelInterpolator->IsInsideBuffer(index))
        modelValue = modelInterpolator->EvaluateAtContinuousIndex(index);
}

} // end of namespace anima
