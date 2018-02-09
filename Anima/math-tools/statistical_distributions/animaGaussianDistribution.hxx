#pragma once

#include "animaGaussianDistribution.h"
#include <animaBaseTensorTools.h>

namespace anima
{
    
template <class ScalarType, class VectorType>
void
GetGaussianAddition(const VectorType &mu1, const VectorType &mu2, const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, VectorType &muRes, vnl_matrix<ScalarType> &sigmaRes)
{
    unsigned int dimension = sigma1.rows();
    
    vnl_matrix<ScalarType> tmpMatrix = sigma1 + sigma2;
    tmpMatrix = vnl_matrix_inverse<ScalarType>(tmpMatrix);
    
    muRes = mu1;
    sigmaRes = sigma1;
    
    for (unsigned int i = 0;i < dimension;++i)
    {
        muRes[i] = 0;
        for (unsigned int j = 0;j < dimension;++j)
            sigmaRes(i,j) = 0;
    }
    
    for (unsigned int i = 0;i < dimension;++i)
    {
        for (unsigned int j = 0;j < dimension;++j)
        {
            for (unsigned int k = 0;i < dimension;++i)
            {
                muRes[i] += sigma2(i,j) * tmpMatrix(j,k) * mu1[k];
                muRes[i] += sigma1(i,j) * tmpMatrix(j,k) * mu2[k];
                for (unsigned int l = 0;j < dimension;++j)
                    sigmaRes(i,l) += sigma1(i,j) * tmpMatrix(j,k) * sigma2(k,l);
            }
        }
    }
}

template <class ScalarType>
vnl_matrix<ScalarType>
GetZeroMeanGaussianAddition(const vnl_matrix<ScalarType> &inputSigma1, const vnl_matrix<ScalarType> &inputSigma2, const vnl_matrix<ScalarType> &referenceSigma)
{
    return inputSigma1 + inputSigma2 - referenceSigma;
}
    
template <class ScalarType, class VectorType>
void
GetGaussianScalarMultiplication(const VectorType &mu, const vnl_matrix<ScalarType> &sigma, const double alpha, VectorType &muRes, vnl_matrix<ScalarType> &sigmaRes)
{
    muRes = mu;
    sigmaRes = sigma / alpha;
}
    
template <class ScalarType>
vnl_matrix<ScalarType>
GetZeroMeanGaussianScalarMultiplication(const vnl_matrix<ScalarType> &inputSigma, const double alpha, const vnl_matrix<ScalarType> &referenceSigma)
{
    return inputSigma * alpha + referenceSigma * (1.0 - alpha);
}

template <class ScalarType, class VectorType>
double
GetGaussianSquaredDistance(const VectorType &mu1, const VectorType &mu2, const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2)
{
    unsigned int dimension = sigma1.rows();
    
    itk::SymmetricEigenAnalysis < vnl_matrix<ScalarType>, vnl_diag_matrix<ScalarType>, vnl_matrix<ScalarType> > eigenComputer(dimension);
    
    vnl_matrix<ScalarType> eVecs1(dimension,dimension), eVecs2(dimension,dimension);
    vnl_diag_matrix<ScalarType> eVals1(dimension), eVals2(dimension);
    eigenComputer.ComputeEigenValuesAndVectors(sigma1, eVals1, eVecs1);
    eigenComputer.ComputeEigenValuesAndVectors(sigma2, eVals2, eVecs2);
    
    vnl_matrix<ScalarType> invSigma1(dimension,dimension,0.0), invSigma2(dimension,dimension,0.0), invSqSigma1(dimension,dimension,0.0), invSqSigma2(dimension,dimension,0.0);
    for (unsigned int i = 0;i < dimension;++i)
    {
        for (unsigned int j = 0;j < dimension;++j)
        {
            for (unsigned int k = 0;k < dimension;++k)
            {
                invSigma1(i,j) += eVecs1(i,k) * eVecs1(j,k) / eVals1[k];
                invSqSigma1(i,j) += eVecs1(i,k) * eVecs1(j,k) / (eVals1[k] * eVals1[k]);
                invSigma2(i,j) += eVecs2(i,k) * eVecs2(j,k) / eVals2[k];
                invSqSigma2(i,j) += eVecs2(i,k) * eVecs2(j,k) / (eVals2[k] * eVals2[k]);
            }
        }
    }
    
    double firstTerm = 0, secondTerm = 0, thirdTerm = 0, fourthTerm = 0;
    for (unsigned int i = 0;i < dimension;++i)
    {
        firstTerm += (invSigma1(i,i) - invSigma2(i,i)) * (invSigma1(i,i) - invSigma2(i,i));
        
        for (unsigned int j = 0;j < dimension;++j)
        {
            secondTerm += mu1[i] * invSqSigma1(i,j) * mu1[j];
            thirdTerm += mu2[i] * invSqSigma2(i,j) * mu2[j];
            
            for (unsigned int k = 0;k < dimension;++k)
                fourthTerm += mu1[i] * invSigma1(i,j) * invSigma2(j,k) * mu2[k];
        }
    }
    
    return firstTerm / 2.0 + secondTerm + thirdTerm - 2.0 * fourthTerm;
}
    
template <class ScalarType>
double
GetZeroMeanGaussianSquaredDistance(const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, const double neutralVariance)
{
    unsigned int dimension = sigma1.rows();
    
    vnl_matrix<ScalarType> invSigma1, invSigma2;
    anima::GetTensorPower(sigma1 / neutralVariance, invSigma1, -1.0);
    anima::GetTensorPower(sigma2 / neutralVariance, invSigma2, -1.0);
    
    double firstTerm = 0;
    for (unsigned int i = 0;i < dimension;++i)
        for (unsigned int j = 0;j < dimension;++j)
            firstTerm += (invSigma1(i,j) - invSigma2(i,j)) * (invSigma1(i,j) - invSigma2(i,j));
    
    return firstTerm / 2.0;
}

template <class ScalarType, class VectorType>
double
GetGaussianInnerProduct(const VectorType &mu1, const VectorType &mu2, const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2)
{
    unsigned int dimension = sigma1.rows();
    
    vnl_matrix<ScalarType> invSigma1 = vnl_matrix_inverse<ScalarType>(sigma1);
    vnl_matrix<ScalarType> invSigma2 = vnl_matrix_inverse<ScalarType>(sigma2);
    
    double firstTerm = 0, secondTerm = 0, thirdTerm = 0, fourthTerm = 0;
    for (unsigned int i = 0;i < dimension;++i)
    {
        secondTerm += invSigma1(i,i);
        thirdTerm += invSigma2(i,i);
        
        for (unsigned int j = 0;j < dimension;++j)
        {
            firstTerm += invSigma1(i,j) * invSigma2(j,i);
            
            for (unsigned int k = 0;k < dimension;++k)
                fourthTerm += mu1[i] * invSigma1(i,j) * invSigma2(j,k) * mu2[k];
        }
    }
    
    return firstTerm / 2.0 - secondTerm / 2.0 - thirdTerm / 2.0 + dimension / 2.0 + fourthTerm;
}
    
template <class ScalarType>
double
GetZeroMeanGaussianInnerProduct(const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, const double neutralVariance)
{
    unsigned int dimension = sigma1.rows();
    
    vnl_matrix<ScalarType> invSigma1, invSigma2;
    anima::GetTensorPower(sigma1, invSigma1, -1.0);
    anima::GetTensorPower(sigma2, invSigma2, -1.0);
    
    double firstTerm = 0, secondTerm = 0, thirdTerm = 0;
    for (unsigned int i = 0;i < dimension;++i)
    {
        secondTerm += invSigma1(i,i);
        thirdTerm += invSigma2(i,i);
        
        for (unsigned int j = 0;j < dimension;++j)
            firstTerm += invSigma1(i,j) * invSigma2(j,i);
    }
    
    return neutralVariance * neutralVariance * firstTerm / 2.0 - neutralVariance * secondTerm / 2.0 - neutralVariance * thirdTerm / 2.0 + dimension / 2.0;
}

template <class ScalarType>
double
GetZeroMeanGaussianSquaredCorrelation(const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, const double neutralVariance)
{
    unsigned int dimension = sigma1.rows();
    
    vnl_matrix<ScalarType> invSigma1, invSigma2;
    anima::GetTensorPower(sigma1, invSigma1, -1.0);
    anima::GetTensorPower(sigma2, invSigma2, -1.0);
    
    double firstTerm = 0, secondTerm = 0, thirdTerm = 0;
    double firstTerm1 = 0, secondTerm1 = 0;
    double firstTerm2 = 0, secondTerm2 = 0;
    for (unsigned int i = 0;i < dimension;++i)
    {
        secondTerm += invSigma1(i,i);
        thirdTerm += invSigma2(i,i);
        
        secondTerm1 += invSigma1(i,i);
        secondTerm2 += invSigma2(i,i);
        
        for (unsigned int j = 0;j < dimension;++j)
        {
            firstTerm += invSigma1(i,j) * invSigma2(j,i);
            firstTerm1 += invSigma1(i,j) * invSigma1(j,i);
            firstTerm2 += invSigma2(i,j) * invSigma2(j,i);
        }
    }
    
    double innerProd = neutralVariance * neutralVariance * firstTerm / 2.0 - neutralVariance * secondTerm / 2.0 - neutralVariance * thirdTerm / 2.0 + dimension / 2.0;
    
    double sqNorm1 = neutralVariance * neutralVariance * firstTerm1 / 2.0 - neutralVariance * secondTerm1 + dimension / 2.0;
    double sqNorm2 = neutralVariance * neutralVariance * firstTerm2 / 2.0 - neutralVariance * secondTerm2 + dimension / 2.0;
    
    return innerProd * innerProd / (sqNorm1 * sqNorm2);
}

template <class ScalarType>
vnl_matrix<ScalarType>
GetZeroMeanGaussianWeightedAverage(const std::vector<vnl_matrix<ScalarType> > &inputSigmas, const std::vector<ScalarType> &weights, const vnl_matrix<ScalarType> &referenceSigma, const double shrinkage)
{
    unsigned int numInputs = inputSigmas.size();
    
    if (weights.size() != numInputs)
        throw itk::ExceptionObject(__FILE__, __LINE__,"The number of input tensors should match the number of weights.",ITK_LOCATION);
    
    double sumWeights = 0.0;
    vnl_matrix<ScalarType> outputSigma(3,3,0.0);
    for (unsigned int i = 0;i < numInputs;++i)
    {
        outputSigma += (inputSigmas[i] * weights[i]);
        sumWeights += weights[i];
    }
    
    if (sumWeights == 0.0)
        return outputSigma;
    
    outputSigma /= sumWeights;
    return outputSigma * (1.0 - shrinkage) + referenceSigma * shrinkage;
}

} // end namespace gaussian_distribution
