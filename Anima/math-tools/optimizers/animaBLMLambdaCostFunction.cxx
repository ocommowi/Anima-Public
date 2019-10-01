#include "animaBLMLambdaCostFunction.h"
#include <animaMatrixOperations.h>
#include <animaBVLSOptimizer.h>
#include <vnl/algo/vnl_qr.h>

namespace anima
{

BLMLambdaCostFunction::MeasureType
BLMLambdaCostFunction::GetValue(const ParametersType &parameters) const
{
    unsigned int nbParams = m_LowerBoundsPermutted.size();

    ParametersType pPermutted(nbParams), pPermuttedShrunk(m_JRank);
    pPermutted.fill(0.0);

    // Solve (JtJ + lambda d^2) x = - Jt r, for a given lambda as parameter
    // Use the fact that pi^T D pi and pi a permutation matrix when D diagonal is diagonal, is equivalent to d[pivotVector[i]] in vector form

    if (parameters[0] > 0.0)
    {
        // Compute QR solution for any lambda > 0.0
        for (unsigned int i = 0;i < nbParams;++i)
            m_WorkMatrix(m_JRank + i,i) = std::sqrt(parameters[0]) * m_DValues[m_PivotVector[i]];

        pPermutted = vnl_qr <double> (m_WorkMatrix).solve(m_WResiduals);
        bool inBounds = this->CheckSolutionIsInBounds(pPermutted);

        if (!inBounds)
        {
            anima::BVLSOptimizer::Pointer bvlsOpt = anima::BVLSOptimizer::New();
            bvlsOpt->SetDataMatrix(m_WorkMatrix);
            bvlsOpt->SetPoints(m_WResiduals);
            bvlsOpt->SetLowerBounds(m_LowerBoundsPermutted);
            bvlsOpt->SetUpperBounds(m_UpperBoundsPermutted);
            bvlsOpt->StartOptimization();

            pPermutted = bvlsOpt->GetCurrentPosition();
        }
    }
    else
    {
        // Compute simpler solution if tested parameter is zero
        // (solver uses only square rank subpart of work matrix, and rank first wresiduals)
        anima::UpperTriangularSolver(m_ZeroWorkMatrix,m_ZeroWResiduals,pPermuttedShrunk,m_JRank);
        bool inBounds = this->CheckSolutionIsInBounds(pPermuttedShrunk);

        if (!inBounds)
        {
            anima::BVLSOptimizer::Pointer bvlsOpt = anima::BVLSOptimizer::New();
            bvlsOpt->SetDataMatrix(m_ZeroWorkMatrix);
            bvlsOpt->SetPoints(m_ZeroWResiduals);
            bvlsOpt->SetLowerBounds(m_LowerBoundsPermutted);
            bvlsOpt->SetUpperBounds(m_UpperBoundsPermutted);
            bvlsOpt->StartOptimization();

            pPermuttedShrunk = bvlsOpt->GetCurrentPosition();
        }

        for (unsigned int i = 0;i < m_JRank;++i)
            pPermutted[i] = pPermuttedShrunk[i];

        for (unsigned int i = m_JRank;i < nbParams;++i)
        {
            pPermutted[i] = std::min(pPermutted[i],m_UpperBoundsPermutted[i]);
            pPermutted[i] = std::max(pPermutted[i],m_LowerBoundsPermutted[i]);
        }
    }

    double phiNorm = 0.0;

    m_LastSolutionVector.set_size(nbParams);
    m_LastSolutionVector.fill(0.0);

    for (unsigned int i = 0;i < nbParams;++i)
    {
        m_LastSolutionVector[i] = pPermutted[m_InversePivotVector[i]];
        double phiVal = m_DValues[i] * pPermutted[m_InversePivotVector[i]];
        phiNorm += phiVal * phiVal;
    }

    phiNorm = std::sqrt(phiNorm);
    return phiNorm - m_DeltaParameter;
}

bool BLMLambdaCostFunction::CheckSolutionIsInBounds(ParametersType &solutionVector) const
{
    unsigned int rank = solutionVector.size();
    for (unsigned int i = 0;i < rank;++i)
    {
        if (solutionVector[i] < m_LowerBoundsPermutted[i])
            return false;

        if (solutionVector[i] > m_UpperBoundsPermutted[i])
            return false;
    }

    return true;
}

void
BLMLambdaCostFunction::GetDerivative(const ParametersType &parameters, DerivativeType &derivative) const
{
    // Voluntarily null because of BVLS
    itkExceptionMacro("No available derivative in BLMLambdaCostFunction");
}

void
BLMLambdaCostFunction::SetWorkMatricesAndVectorsFromQRDerivative(vnl_matrix <double> &qrDerivative,
                                                                 ParametersType &qtResiduals, unsigned int rank)
{
    unsigned int nbParams = qrDerivative.cols();
    unsigned int numLinesWorkMatrix = rank + nbParams;

    m_ZeroWorkMatrix.set_size(rank,rank);
    m_ZeroWorkMatrix.fill(0.0);
    m_ZeroWResiduals.set_size(rank);
    m_ZeroWResiduals.fill(0.0);

    m_WorkMatrix.set_size(numLinesWorkMatrix,nbParams);
    m_WorkMatrix.fill(0.0);
    m_WResiduals.set_size(numLinesWorkMatrix);
    m_WResiduals.fill(0.0);

    for (unsigned int i = 0;i < rank;++i)
    {
        for (unsigned int j = i;j < rank;++j)
        {
            m_ZeroWorkMatrix(i,j) = qrDerivative(i,j);
            m_WorkMatrix(i,j) = qrDerivative(i,j);
        }

        for (unsigned int j = rank;j < nbParams;++j)
            m_WorkMatrix(i,j) = qrDerivative(i,j);

        m_WResiduals[i] = - qtResiduals[i];
        m_ZeroWResiduals[i] = - qtResiduals[i];
    }
}

} // end namespace anima