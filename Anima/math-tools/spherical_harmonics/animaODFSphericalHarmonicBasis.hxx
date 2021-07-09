#pragma once
#include "animaODFSphericalHarmonicBasis.h"

namespace anima
{

template <class T>
double
ODFSphericalHarmonicBasis::
getValueAtPosition(const T &coefficients, double theta, double phi)
{
    double resVal = 0;
    std::complex <double> tmpVal;

    for (unsigned int k = 0;k <= m_LOrder;k += 2)
    {
        unsigned int kIndexCoef = k * (k + 1) / 2;
        unsigned int kIndexSH = k * k / 4;

        for (unsigned int m = 0;m <= k;++m)
        {
            tmpVal = m_SphericalHarmonics[kIndexSH + m].Value(theta,phi);

            if (m != 0)
            {
                // Part for m > 0
                resVal += sqrt(2.0) * coefficients[kIndexCoef + m] * imag(tmpVal);

                // Part for m < 0
                if (m % 2 == 0)
                    resVal += sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
                else
                    resVal -= sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
            }
            else
                resVal += coefficients[kIndexCoef]*real(tmpVal);
        }
    }

    return resVal;
}

template <class T>
itk::VariableLengthVector <T>
ODFSphericalHarmonicBasis::
GetSampleValues(itk::VariableLengthVector <T> &data,
                std::vector < std::vector <double> > &m_SampleDirections)
{
    itk::VariableLengthVector <T> resVal(m_SampleDirections.size());

    for (unsigned int i = 0;i < m_SampleDirections.size();++i)
        resVal[i] = this->getValueAtPosition(data,m_SampleDirections[i][0],m_SampleDirections[i][1]);

    return resVal;
}

template <class T>
double
ODFSphericalHarmonicBasis::
getThetaFirstDerivativeValueAtPosition(const T &coefficients, double theta, double phi)
{
    double resVal = 0;
    std::complex <double> tmpVal;

    for (unsigned int k = 0;k <= m_LOrder;k += 2)
    {
        unsigned int kIndexCoef = k * (k + 1) / 2;
        unsigned int kIndexSH = k * k / 4;

        for (unsigned int m = 0;m <= k;++m)
        {
            tmpVal = m_SphericalHarmonics[kIndexSH + m].getThetaFirstDerivative(theta,phi);

            if (m != 0)
            {
                // Part for m > 0
                resVal += sqrt(2.0) * coefficients[kIndexCoef + m] * imag(tmpVal);

                // Part for m < 0
                if (m % 2 == 0)
                    resVal += sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
                else
                    resVal -= sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
            }
            else
                resVal += coefficients[kIndexCoef]*real(tmpVal);
        }
    }

    return resVal;
}

template <class T>
double
ODFSphericalHarmonicBasis::
getPhiFirstDerivativeValueAtPosition(const T &coefficients, double theta, double phi)
{
    double resVal = 0;
    std::complex <double> tmpVal;

    for (unsigned int k = 0;k <= m_LOrder;k += 2)
    {
        unsigned int kIndexCoef = k * (k + 1) / 2;
        unsigned int kIndexSH = k * k / 4;

        for (unsigned int m = 0;m <= k;++m)
        {
            tmpVal = m_SphericalHarmonics[kIndexSH + m].getPhiFirstDerivative(theta,phi);

            if (m != 0)
            {
                // Part for m > 0
                resVal += sqrt(2.0) * coefficients[kIndexCoef + m] * imag(tmpVal);

                // Part for m < 0
                if (m % 2 == 0)
                    resVal += sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
                else
                    resVal -= sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
            }
            else
                resVal += coefficients[kIndexCoef]*real(tmpVal);
        }
    }

    return resVal;
}

template <class T>
double
ODFSphericalHarmonicBasis::
getThetaSecondDerivativeValueAtPosition(const T &coefficients, double theta, double phi)
{
    double resVal = 0;
    std::complex <double> tmpVal;

    for (unsigned int k = 0;k <= m_LOrder;k += 2)
    {
        unsigned int kIndexCoef = k * (k + 1) / 2;
        unsigned int kIndexSH = k * k / 4;

        for (unsigned int m = 0;m <= k;++m)
        {
            tmpVal = m_SphericalHarmonics[kIndexSH + m].getThetaSecondDerivative(theta,phi);

            if (m != 0)
            {
                // Part for m > 0
                resVal += sqrt(2.0) * coefficients[kIndexCoef + m] * imag(tmpVal);

                // Part for m < 0
                if (m % 2 == 0)
                    resVal += sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
                else
                    resVal -= sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
            }
            else
                resVal += coefficients[kIndexCoef]*real(tmpVal);
        }
    }

    return resVal;
}

template <class T>
double
ODFSphericalHarmonicBasis::
getThetaPhiDerivativeValueAtPosition(const T &coefficients, double theta, double phi)
{
    double resVal = 0;
    std::complex <double> tmpVal;

    for (unsigned int k = 0;k <= m_LOrder;k += 2)
    {
        unsigned int kIndexCoef = k * (k + 1) / 2;
        unsigned int kIndexSH = k * k / 4;

        for (unsigned int m = 0;m <= k;++m)
        {
            tmpVal = m_SphericalHarmonics[kIndexSH + m].getThetaPhiDerivative(theta,phi);

            if (m != 0)
            {
                // Part for m > 0
                resVal += sqrt(2.0) * coefficients[kIndexCoef + m] * imag(tmpVal);

                // Part for m < 0
                if (m % 2 == 0)
                    resVal += sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
                else
                    resVal -= sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
            }
            else
                resVal += coefficients[kIndexCoef]*real(tmpVal);
        }
    }

    return resVal;
}

template <class T>
double
ODFSphericalHarmonicBasis::
getPhiSecondDerivativeValueAtPosition(const T &coefficients, double theta, double phi)
{
    double resVal = 0;
    std::complex <double> tmpVal;

    for (unsigned int k = 0;k <= (int)m_LOrder;k += 2)
    {
        unsigned int kIndexCoef = k * (k + 1) / 2;
        unsigned int kIndexSH = k * k / 4;

        for (unsigned int m = 0;m <= k;++m)
        {
            tmpVal = m_SphericalHarmonics[kIndexSH + m].getPhiSecondDerivative(theta,phi);

            if (m != 0)
            {
                // Part for m > 0
                resVal += sqrt(2.0) * coefficients[kIndexCoef + m] * imag(tmpVal);

                // Part for m < 0
                if (m % 2 == 0)
                    resVal += sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
                else
                    resVal -= sqrt(2.0) * coefficients[kIndexCoef - m] * real(tmpVal);
            }
            else
                resVal += coefficients[kIndexCoef]*real(tmpVal);
        }
    }

    return resVal;
}

template <class T>
double
ODFSphericalHarmonicBasis::
getCurvatureAtPosition(const T &coefficients, double theta, double phi)
{
    // Taken from Bloy and Verma, simplified to the maximum (this supposes we are actually at an extremum of the odf)
    double odfValue = this->getValueAtPosition(coefficients,theta,phi);
    double sqSinTheta = sin(theta) * sin(theta);
    if (sqSinTheta <= 1.0e-16)
        sqSinTheta = 1.0e-16;

    double denom = 2.0 * odfValue * odfValue * sqSinTheta;

    double num = 2.0 * odfValue * sqSinTheta;
    num -= sqSinTheta * this->getThetaSecondDerivativeValueAtPosition(coefficients,theta,phi) +
    this->getPhiSecondDerivativeValueAtPosition(coefficients,theta,phi);

    return num / denom;
}

template <class T>
double
ODFSphericalHarmonicBasis::
getAbsoluteUpperBound(const T &coefficients)
{
    double resVal = 0;
    double tmpVal;

    for (unsigned int k = 0;k <= (int)m_LOrder;k += 2)
    {
        unsigned int kIndexCoef = k * (k + 1) / 2;
        unsigned int kIndexSH = k * k / 4;

        for (unsigned int m = 0;m <= k;++m)
        {
            tmpVal = m_SphericalHarmonics[kIndexSH + m].getSHAbsoluteBound();

            if (m != 0)
            {
                // Part for m > 0
                resVal += sqrt(2.0) * std::abs(coefficients[kIndexCoef + m]) * tmpVal;

                // Part for m < 0
                resVal += sqrt(2.0) * std::abs(coefficients[kIndexCoef - m]) * tmpVal;
            }
            else
                resVal += std::abs(coefficients[kIndexCoef]) * tmpVal;
        }
    }

    return resVal;
}

template <class T1, class T2>
void
ODFSphericalHarmonicBasis::
getDirectionSampleFromODF(const T1 &coefficients, T2 &outputDirection, std::mt19937 &generator)
{
    std::uniform_real_distribution<double> uniDbl(0.0,1.0);
    double upperBound = this->getAbsoluteUpperBound(coefficients);

    bool continueLoop = true;
    double theta, phi;
    while (continueLoop)
    {
        theta = std::acos(1.0 - 2.0 * uniDbl(generator));
        phi = 2.0 * M_PI * uniDbl(generator);

        double shBValue = this->getValueAtPosition(coefficients, theta, phi);

        double thr = uniDbl(generator) * upperBound;
        if (shBValue < thr)
            continueLoop = false;
    }

    outputDirection[0] = std::sin(theta) * std::cos(phi);
    outputDirection[1] = std::sin(theta) * std::sin(phi);
    outputDirection[2] = std::cos(theta);
}

} // end of namespace anima
