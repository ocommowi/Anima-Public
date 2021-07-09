#include "animaODFProbabilisticTractographyImageFilter.h"
#include <cmath>
#include <random>

#include <animaODFMaximaCostFunction.h>
#include <animaNLOPTOptimizers.h>

#include <animaVectorOperations.h>
#include <animaMatrixOperations.h>
#include <animaDistributionSampling.h>
#include <animaVMFDistribution.h>
#include <animaWatsonDistribution.h>

namespace anima
{

ODFProbabilisticTractographyImageFilter::ODFProbabilisticTractographyImageFilter()
    : BaseProbabilisticTractographyImageFilter()
{
    m_ODFSHOrder = 4;
    m_GFAThreshold = 0.1;

    m_ODFSHBasis = NULL;

    this->SetModelDimension(15);
}

ODFProbabilisticTractographyImageFilter::~ODFProbabilisticTractographyImageFilter()
{
    if (m_ODFSHBasis)
        delete m_ODFSHBasis;
}

void ODFProbabilisticTractographyImageFilter::PrepareTractography()
{
    // Call base preparation
    BaseProbabilisticTractographyImageFilter::PrepareTractography();

    m_ODFSHOrder = std::round(-1.5 + 0.5 * std::sqrt(8 * this->GetInputModelImage()->GetNumberOfComponentsPerPixel() + 1));
    this->SetModelDimension((m_ODFSHOrder + 1)*(m_ODFSHOrder + 2)/2);

    // Initialize estimation matrices for Aganj et al based estimation
    if (m_ODFSHBasis)
        delete m_ODFSHBasis;

    m_ODFSHBasis = new anima::ODFSphericalHarmonicBasis(m_ODFSHOrder);
}

ODFProbabilisticTractographyImageFilter::Vector3DType
ODFProbabilisticTractographyImageFilter::ProposeNewDirection(Vector3DType &oldDirection, VectorType &modelValue,
                                                             std::mt19937 &random_generator, unsigned int threadId)
{
    Vector3DType resVec(0.0);
    bool is2d = (this->GetInputModelImage()->GetLargestPossibleRegion().GetSize()[2] == 1);

    m_ODFSHBasis->getDirectionSampleFromODF(modelValue,resVec,random_generator);

    if (is2d)
    {
        resVec[InputModelImageType::ImageDimension - 1] = 0;
        resVec.Normalize();
    }

    if (anima::ComputeScalarProduct(oldDirection, resVec) < 0)
        resVec *= -1;

    return resVec;
}

bool ODFProbabilisticTractographyImageFilter::CheckModelProperties(double estimatedB0Value, double estimatedNoiseValue, VectorType &modelValue, unsigned int threadId)
{
    if (estimatedB0Value < 50.0)
        return false;

    bool isModelNull = true;
    for (unsigned int j = 0;j < this->GetModelDimension();++j)
    {
        if (modelValue[j] != 0)
        {
            isModelNull = false;
            break;
        }
    }

    if (isModelNull)
        return false;

    double fractionalAnisotropy = this->GetGeneralizedFractionalAnisotropy(modelValue);
    if (fractionalAnisotropy < m_GFAThreshold)
        return false;

    return true;
}

double ODFProbabilisticTractographyImageFilter::ComputeLogWeightUpdate(double b0Value, double noiseValue, Vector3DType &previousDirection,
                                                                       Vector3DType &newDirection, VectorType &previousModelValue, VectorType &modelValue,
                                                                       unsigned int threadId)
{
    // Prior is a Watson PDF
    double log_prior = anima::safe_log(anima::EvaluateWatsonPDF(newDirection, previousDirection, this->GetKappaOfPriorDistribution()));

    // Proposal is the probability to get the sampled new direction from the previous model. We assume the ODF integral is 1
    Vector3DType sphDirection;
    anima::TransformCartesianToSphericalCoordinates(newDirection,sphDirection);
    double log_proposal = m_ODFSHBasis->getValueAtPosition(previousModelValue,sphDirection[0],sphDirection[1]);

    // Log likelihood : simulate signal fron Aganj et al for arrival model
    // and a TOD from input direction
    double log_likelihood = 0.0;

    VectorType todValue(this->GetModelDimension());
    VectorType modelSignalValue(this->GetModelDimension());
    unsigned int pos = 0;
    for (unsigned int l = 0;l < m_ODFSHOrder;l += 2)
    {
        double pjValNum = 1, pjValDenom = 1, pjVal;

        for (unsigned int k = 2;k <= l;k += 2)
        {
            pjValNum *= (k-1);
            pjValDenom *= k;
        }

        pjVal = pjValNum/pjValDenom;
        if (l/2 % 2 != 0)
            pjVal *= -1;

        for (int m = -l;m <= l;++m)
        {
            todValue[pos] = m_ODFSHBasis->getNthSHValueAtPosition(l,m,sphDirection[0],sphDirection[1]);
            todValue[pos] /= (2.0 * M_PI * pjVal);
            modelSignalValue[pos] = modelValue[pos] / (2.0 * M_PI * pjVal);
            ++pos;
        }
    }

    unsigned int numGradientImages = this->GetNumberOfGradientDirections();
    Vector3DType gradient;

    for (unsigned int i = 0;i < numGradientImages;++i)
    {
        double bvalue = this->GetBValueItem(i);
        if (bvalue == 0)
        {
            log_likelihood -= 0.5 * (std::log(noiseValue) + std::log(2.0 * M_PI));
            continue;
        }

        gradient = this->GetDiffusionGradient(i);
        anima::TransformCartesianToSphericalCoordinates(gradient,sphDirection);
        double todSignal = m_ODFSHBasis->getValueAtPosition(todValue,sphDirection[0],sphDirection[1]);
        double modelSignal = m_ODFSHBasis->getValueAtPosition(modelSignalValue,sphDirection[0],sphDirection[1]);

        double diffValue = b0Value * (todSignal - modelSignal);
        double logNoiseGaussValue = 0.5 * (std::log(noiseValue) + std::log(2.0 * M_PI) + diffValue * diffValue / noiseValue);

        log_likelihood -= logNoiseGaussValue;
    }

    double resVal = log_likelihood + log_prior - log_proposal;
    return resVal;
}

void ODFProbabilisticTractographyImageFilter::ComputeModelValue(InterpolatorPointer &modelInterpolator, ContinuousIndexType &index,
                                                                VectorType &modelValue)
{
    modelValue.SetSize(this->GetModelDimension());
    modelValue.Fill(0.0);

    if (modelInterpolator->IsInsideBuffer(index))
        modelValue = modelInterpolator->EvaluateAtContinuousIndex(index);
}

double ODFProbabilisticTractographyImageFilter::GetGeneralizedFractionalAnisotropy(VectorType &modelValue)
{
    double sumSquares = 0;
    for (unsigned int i = 0;i < this->GetModelDimension();++i)
        sumSquares += modelValue[i]*modelValue[i];

    return std::sqrt(1.0 - modelValue[0]*modelValue[0]/sumSquares);
}

} // end of namespace anima
