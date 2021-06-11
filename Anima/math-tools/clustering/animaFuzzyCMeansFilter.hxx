#pragma once
#include "animaFuzzyCMeansFilter.h"

#include <iostream>
#include <cmath>
#include <limits>

#include <animaLogExpMapsUnitSphere.h>
#include <animaGaussianDistribution.h>
#include <animaBaseTensorTools.h>

namespace anima
{

template <class ScalarType>
FuzzyCMeansFilter <ScalarType>
::FuzzyCMeansFilter()
{
    m_ClassesMembership.clear();
    m_Centroids.clear();
    m_InputData.clear();

    m_NbClass = 0;
    m_NbInputs = 0;
    m_NDim = 0;
    m_MaxIterations = 100;

    m_Verbose = true;
    m_SpectralInitialization = true;
    m_AverageType = Euclidean;

    m_RelStopCriterion = 1.0e-4;
    m_MValue = 2;
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::SetInputData(DataHolderType &data)
{
    if (data.size() == 0)
        return;

    m_InputData = data;
    m_NbInputs = m_InputData.size();
    m_NDim = m_InputData[0].size();
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::Update()
{
    if (m_NbClass > m_NbInputs)
        throw itk::ExceptionObject(__FILE__,__LINE__,"More classes than inputs...",ITK_LOCATION);

    m_DataWeights.resize(m_NbInputs);
    std::fill(m_DataWeights.begin(),m_DataWeights.end(),1.0 / m_NbInputs);

    InitializeCMeansFromData();
    DataHolderType oldMemberships = m_ClassesMembership;

    unsigned int itncount = 0;
    bool continueLoop = true;

    while ((itncount < m_MaxIterations)&&(continueLoop))
    {
        itncount++;

        if (m_Verbose)
            std::cout << "Iteration " << itncount << "..." << std::endl;

        ComputeCentroids();
        UpdateMemberships();

        continueLoop = !endConditionReached(oldMemberships);
        oldMemberships = m_ClassesMembership;
    }
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::ComputeCentroids()
{
    if (m_PowMemberships.size() != m_NbInputs)
        m_PowMemberships.resize(m_NbInputs);

    for (unsigned int i = 0;i < m_NbInputs;++i)
    {
        if (m_PowMemberships[i].size() != m_NbClass)
            m_PowMemberships[i].resize(m_NbClass);

        for (unsigned int j = 0;j < m_NbClass;++j)
            m_PowMemberships[i][j] = std::pow(m_ClassesMembership[i][j],m_MValue);
    }

    if (m_TmpVector.size() != m_NDim)
        m_TmpVector.resize(m_NDim);

    if (m_TmpWeights.size() != m_NbInputs)
        m_TmpWeights.resize(m_NbInputs);

    for (unsigned int i = 0;i < m_NbClass;++i)
    {
        switch (m_AverageType)
        {
            case ApproximateSpherical:
            case Spherical:
                this->ComputeSphericalCentroid(i);
                break;

            case BayesTensors:
                this->ComputeBayesCentroid(i);
                break;

            case Euclidean:
            default:
                this->ComputeEuclideanCentroid(i);
                break;
        }
    }
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::ComputeBayesCentroid(unsigned int i)
{
    double sumPowMemberShips = 0;
    m_TmpVector.resize(m_NDim);

    for (unsigned int j = 0;j < m_NbInputs;++j)
        sumPowMemberShips += m_DataWeights[j] * m_PowMemberships[j][i];

    itk::VariableLengthVector <ScalarType> tmpVec(m_NDim);
    unsigned int tensDim = std::floor(std::sqrt(8 * m_NDim + 1) - 1) / 2;
    vnl_matrix<double> referenceMatrix(tensDim,tensDim,0.0);
    referenceMatrix.fill_diagonal(3e-3);
    vnl_matrix <double> tmpMatrix, meanMatrix(referenceMatrix);

    for (unsigned int j = 0;j < m_NbInputs;++j)
    {
        for (unsigned int k = 0;k < m_NDim;++k)
            tmpVec[k] = m_InputData[j][k];

        anima::GetTensorFromVectorRepresentation(tmpVec,tmpMatrix,tensDim,false);
        tmpMatrix = anima::GetZeroMeanGaussianScalarMultiplication(tmpMatrix, m_DataWeights[j] * m_PowMemberships[j][i] / sumPowMemberShips, referenceMatrix);
        meanMatrix = anima::GetZeroMeanGaussianAddition(tmpMatrix, meanMatrix, referenceMatrix);
    }

    anima::GetVectorRepresentation(meanMatrix,tmpVec,m_NDim,false);

    for (unsigned int k = 0;k < m_NDim;++k)
        m_TmpVector[k] = tmpVec[k];

    m_Centroids[i] = m_TmpVector;
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::ComputeEuclideanCentroid(unsigned int i)
{
    double sumPowMemberShips = 0;
    m_TmpVector.resize(m_NDim);

    std::fill(m_TmpVector.begin(),m_TmpVector.end(),0.0);

    for (unsigned int j = 0;j < m_NbInputs;++j)
    {
        sumPowMemberShips += m_DataWeights[j] * m_PowMemberships[j][i];
        for (unsigned int k = 0;k < m_NDim;++k)
            m_TmpVector[k] += m_DataWeights[j] * m_PowMemberships[j][i] * m_InputData[j][k];
    }

    for (unsigned int k = 0;k < m_NDim;++k)
        m_TmpVector[k] /= sumPowMemberShips;

    m_Centroids[i] = m_TmpVector;
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::ComputeSphericalCentroid(unsigned int i)
{
    double sumPowMemberShips = 0;
    m_TmpVector.resize(m_NDim);

    std::fill(m_TmpVector.begin(),m_TmpVector.end(),0.0);

    for (unsigned int j = 0;j < m_NbInputs;++j)
    {
        sumPowMemberShips += m_DataWeights[j] * m_PowMemberships[j][i];
        for (unsigned int k = 0;k < m_NDim;++k)
            m_TmpVector[k] += m_DataWeights[j] * m_PowMemberships[j][i] * m_InputData[j][k];
    }

    for (unsigned int k = 0;k < m_NDim;++k)
        m_TmpVector[k] /= sumPowMemberShips;

    if (m_AverageType == ApproximateSpherical)
    {
        double tmpSum = 0;
        for (unsigned int k = 0;k < m_NDim;++k)
            tmpSum += m_TmpVector[k] * m_TmpVector[k];

        tmpSum = std::sqrt(tmpSum);
        for (unsigned int k = 0;k < m_NDim;++k)
            m_TmpVector[k] /= tmpSum;

        m_Centroids[i] = m_TmpVector;
    }
    else
    {
        double tmpSum = 0;
        for (unsigned int k = 0;k < m_NDim;++k)
            tmpSum += m_TmpVector[k] * m_TmpVector[k];

        tmpSum = std::sqrt(tmpSum);
        for (unsigned int k = 0;k < m_NDim;++k)
            m_TmpVector[k] /= tmpSum;

        for (unsigned int k = 0;k < m_NbInputs;++k)
            m_TmpWeights[k] = m_DataWeights[k] * m_PowMemberships[k][i];

        anima::ComputeSphericalCentroid(m_InputData,m_Centroids[i],m_TmpVector,m_TmpWeights,&m_WorkLogVector,&m_WorkVector);
    }
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::UpdateMemberships()
{
    long double powFactor = 1.0/(m_MValue - 1.0);
    m_DistancesPointsCentroids.resize(m_NbClass);

    for (unsigned int i = 0;i < m_NbInputs;++i)
    {
        unsigned int minClassIndex = 0;
        bool nullDistance = false;
        for (unsigned int j = 0;j < m_NbClass;++j)
        {
            m_DistancesPointsCentroids[j] = computeDistance(m_InputData[i],m_Centroids[j]);

            if (m_DistancesPointsCentroids[j] <= 0)
            {
                nullDistance = true;
                minClassIndex = j;
                break;
            }
        }

        if (nullDistance)
        {
            for (unsigned int j = 0;j < m_NbClass;++j)
                m_ClassesMembership[i][j] = 0;

            m_ClassesMembership[i][minClassIndex] = 1.0;
            continue;
        }

        for (unsigned int j = 0;j < m_NbClass;++j)
        {
            long double tmpVal = 0;

            if (m_MValue == 2.0)
            {
                for (unsigned int k = 0;k < m_NbClass;++k)
                    tmpVal += m_DistancesPointsCentroids[j] / m_DistancesPointsCentroids[k];
            }
            else
            {
                for (unsigned int k = 0;k < m_NbClass;++k)
                    tmpVal += std::pow(m_DistancesPointsCentroids[j] / m_DistancesPointsCentroids[k],powFactor);
            }

            m_ClassesMembership[i][j] = 1.0 / tmpVal;
        }
    }
}

template <class ScalarType>
bool
FuzzyCMeansFilter <ScalarType>
::endConditionReached(DataHolderType &oldMemberships)
{
    double absDiff = 0;

    for (unsigned int i = 0;i < m_NbInputs;++i)
    {
        for (unsigned int j = 0;j < m_NbClass;++j)
        {
            double testValue = std::abs(oldMemberships[i][j] - m_ClassesMembership[i][j]);
            if (testValue > absDiff)
                absDiff = testValue;
        }
    }

    if (absDiff > m_RelStopCriterion)
        return false;
    else
        return true;
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::InitializeCMeansFromData()
{
    m_Centroids.resize(m_NbClass);
    m_ClassesMembership.resize(m_NbInputs);
    VectorType tmpVec(m_NbClass,0);

    if (!m_SpectralInitialization)
    {
        for (unsigned int i = 0;i < m_NbClass;++i)
            m_Centroids[i] = m_InputData[i];

        double fixVal = 0.95;
        for (unsigned int i = 0;i < m_NbInputs;++i)
        {
            unsigned int tmp = i % m_NbClass;
            m_ClassesMembership[i] = tmpVec;
            m_ClassesMembership[i][tmp] = fixVal;
            for (unsigned j = 0;j < m_NbClass;++j)
            {
                if (j != tmp)
                    m_ClassesMembership[i][j] = (1.0 - fixVal)/(m_NbClass - 1.0);
            }
        }
    }
    else
    {
        m_Centroids[0] = m_InputData[0];
        std::vector <unsigned int> alreadyIn(m_NbClass,0);

        for (unsigned int i = 1;i < m_NbClass;++i)
        {
            double minCrossProd = std::numeric_limits <double>::max();
            unsigned int minIndex = 0;
            for (unsigned int j = 0;j < m_NbInputs;++j)
            {
                bool useIt = true;
                for (unsigned int k = 0;k < i;++k)
                {
                    if (alreadyIn[k] == j)
                    {
                        useIt = false;
                        break;
                    }
                }

                if (useIt)
                {
                    double maxCrossProd = 0;
                    for (unsigned int l = 0;l < i;++l)
                    {
                        double crossProd = 0;
                        for (unsigned int k = 0;k < m_NDim;++k)
                            crossProd += m_InputData[j][k]*m_Centroids[l][k];

                        if (crossProd > maxCrossProd)
                            maxCrossProd = crossProd;
                    }

                    if (maxCrossProd < minCrossProd)
                    {
                        minCrossProd = maxCrossProd;
                        minIndex = j;
                    }
                }
            }

            m_Centroids[i] = m_InputData[minIndex];
            alreadyIn[i] = minIndex;
        }

        //Centroids initialized, now compute memberships
        for (unsigned int i = 0;i < m_NbInputs;++i)
            m_ClassesMembership[i] = tmpVec;

        this->UpdateMemberships();
    }
}

template <class ScalarType>
void
FuzzyCMeansFilter <ScalarType>
::InitializeClassesMemberships(DataHolderType &classM)
{
    if (classM.size() == m_NbInputs)
    {
        m_ClassesMembership.resize(m_NbInputs);

        for (unsigned int i = 0;i < m_NbInputs;++i)
            m_ClassesMembership[i] = classM[i];
    }
}

template <class ScalarType>
long double
FuzzyCMeansFilter <ScalarType>
::computeDistance(VectorType &vec1, VectorType &vec2)
{
    long double resVal = 0;

    switch (m_AverageType)
    {
        case Spherical:
        case ApproximateSpherical:
        {
            long double dotProd = 0;
            for (unsigned int i = 0;i < m_NDim;++i)
                dotProd += vec1[i]*vec2[i];

            if (dotProd > 1)
                dotProd = 1;

            resVal = std::acos(dotProd) * std::acos(dotProd);
            break;
        }

        case BayesTensors:
        {
            unsigned int tensDim = std::floor(std::sqrt(8 * m_NDim + 1) - 1) / 2;
            vnl_matrix <ScalarType> firstTensor, secondTensor;
            itk::VariableLengthVector <ScalarType> tmpVec(m_NDim);

            for (unsigned int i = 0;i < m_NDim;++i)
                tmpVec[i] = vec1[i];
            anima::GetTensorFromVectorRepresentation(tmpVec,firstTensor,tensDim,false);

            for (unsigned int i = 0;i < m_NDim;++i)
                tmpVec[i] = vec2[i];
            anima::GetTensorFromVectorRepresentation(tmpVec,secondTensor,tensDim,false);

            resVal = anima::GetZeroMeanGaussianSquaredDistance(firstTensor,secondTensor,1.0 / 3.0e-3);
            break;
        }

        case Euclidean:
        default:
            resVal = 0;
            for (unsigned int i = 0;i < m_NDim;++i)
                resVal += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);

            break;
    }

    return resVal;
}

} // end namespace anima
