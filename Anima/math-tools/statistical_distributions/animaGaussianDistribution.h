#pragma once

namespace anima
{

template <class ScalarType, class VectorType>
void
GetGaussianAddition(const VectorType &mu1, const VectorType &mu2, const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, VectorType &muRes, vnl_matrix<ScalarType> &sigmaRes);
    
template <class ScalarType, class VectorType>
vnl_matrix<ScalarType>
GetZeroMeanGaussianAddition(const vnl_matrix<ScalarType> &inputSigma1, const vnl_matrix<ScalarType> &inputSigma2, const vnl_matrix<ScalarType> &referenceSigma);

template <class ScalarType, class VectorType>
void
GetGaussianScalarMultiplication(const VectorType &mu, const vnl_matrix<ScalarType> &sigma, const double alpha, VectorType &muRes, vnl_matrix<ScalarType> &sigmaRes);
    
template <class ScalarType, class VectorType>
vnl_matrix<ScalarType>
GetZeroMeanGaussianScalarMultiplication(const vnl_matrix<ScalarType> &inputSigma, const double alpha, const vnl_matrix<ScalarType> &referenceSigma);

template <class ScalarType, class VectorType>
double
GetGaussianSquaredDistance(const VectorType &mu1, const VectorType &mu2, const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2);

template <class ScalarType, class VectorType>
double
GetZeroMeanGaussianSquaredDistance(const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, const double neutralVariance = 1.0);

template <class ScalarType, class VectorType>
double
GetGaussianInnerProduct(const VectorType &mu1, const VectorType &mu2, const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2);

template <class ScalarType, class VectorType>
double
GetZeroMeanGaussianInnerProduct(const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, const double neutralVariance = 1.0);

template <class ScalarType>
double
GetZeroMeanGaussianSquaredCorrelation(const vnl_matrix<ScalarType> &sigma1, const vnl_matrix<ScalarType> &sigma2, const double neutralVariance);

template <class ScalarType>
vnl_matrix<ScalarType>
GetZeroMeanGaussianWeightedAverage(const std::vector<vnl_matrix<ScalarType> > &inputSigmas, const std::vector<ScalarType> &weights, const vnl_matrix<ScalarType> &referenceSigma, const double shrinkage = 0.0);

} // end of namespace anima

#include "animaGaussianDistribution.hxx"
