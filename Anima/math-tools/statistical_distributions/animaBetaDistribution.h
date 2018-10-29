#pragma once

namespace anima
{
    
template <class ScalarType>
ScalarType GetBetaLogPDF(ScalarType x, ScalarType alpha, ScalarType beta);

template <class ScalarType>
ScalarType GetBetaPDFDerivative(ScalarType x, ScalarType alpha, ScalarType beta);

} // end of namespace anima

#include "animaBetaDistribution.hxx"
