#pragma once
#include "animaGammaDistribution.h"
#include <cmath>
#include <algorithm>

namespace anima
{

template <class ScalarType>
ScalarType GetGammaLogPDF(ScalarType x, ScalarType k, ScalarType theta)
{
    double logGammaConstant = - std::log(std::tgamma(k)) - k * std::log(theta);
    double xClamp = std::max(1.0e-8,x);

    double logPDFValue = logGammaConstant + (k - 1.0) * std::log(xClamp);
    logPDFValue -= xClamp / theta;

    return logPDFValue;
}

template <class ScalarType>
ScalarType GetGammaPDFDerivative(ScalarType x, ScalarType k, ScalarType theta)
{
    double xClamp = std::max(1.0e-8,x);
    double logGammaConstant = - std::log(std::tgamma(k)) - k * std::log(theta);
    double logDiffValue = logGammaConstant + (k - 2.0) * std::log(xClamp) - xClamp / theta;
    double diffValue = (k - 1.0 - xClamp / theta) * std::exp(logDiffValue);

    return diffValue;
}

} // end namespace anima
