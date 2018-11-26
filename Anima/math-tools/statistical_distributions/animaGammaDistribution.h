#pragma once

namespace anima
{
    
template <class ScalarType>
ScalarType GetGammaLogPDF(ScalarType x, ScalarType k, ScalarType theta);

template <class ScalarType>
ScalarType GetGammaPDFDerivative(ScalarType x, ScalarType k, ScalarType theta);

} // end of namespace anima

#include "animaGammaDistribution.hxx"
