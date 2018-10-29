#include <animaBaseMCMCost.h>
#include <animaMCMConstants.h>

namespace anima
{

BaseMCMCost::BaseMCMCost()
{
    m_SigmaSquare = 1;

    m_SmallDelta = anima::DiffusionSmallDelta;
    m_BigDelta = anima::DiffusionBigDelta;

    m_MAPEstimationMode = false;
    m_LogPriorValue = 0.0;
}

} // end namespace anima
