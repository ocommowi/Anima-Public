#pragma once
#include "animaBetaDistribution.h"
#include <cmath>
#include <algorithm>

namespace anima
{

template <class ScalarType>
ScalarType GetBetaLogPDF(ScalarType x, ScalarType alpha, ScalarType beta)
{
    double logBetaConstant = std::log(std::tgamma(alpha + beta)) - std::log(std::tgamma(alpha)) - std::log(std::tgamma(beta));

    double logPDFValue = logBetaConstant + (alpha - 1.0) * std::log(std::max(1.0e-8,x));
    logPDFValue += (beta - 1.0) * std::log(std::max(1.0e-8,1.0 - x));

    return logPDFValue;
}

template <class ScalarType>
ScalarType GetBetaPDFDerivative(ScalarType x, ScalarType alpha, ScalarType beta)
{
    double logBetaConstant = std::log(std::tgamma(alpha + beta)) - std::log(std::tgamma(alpha)) - std::log(std::tgamma(beta));
    double logDiffValue = logBetaConstant + (alpha - 2.0) * std::log(std::max(1.0e-8,x)) + (beta - 2.0) * std::log(std::max(1.0 - x,1.0e-8));
    logDiffValue += std::log((alpha - 1.0) * (1.0 - x) - (beta - 1.0) * x);

    return std::exp(logDiffValue);
}

} // end namespace anima
