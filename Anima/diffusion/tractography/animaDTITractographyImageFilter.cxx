#include "animaDTITractographyImageFilter.h"
#include <itkSymmetricEigenAnalysis.h>
#include <animaVectorOperations.h>

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <animaBaseTensorTools.h>

namespace anima
{

dtiTractographyImageFilter::dtiTractographyImageFilter()
{
    m_StopFAThreshold = 0.1;
    m_StopADCThreshold = 2.0e-3;
}

dtiTractographyImageFilter::~dtiTractographyImageFilter()
{
}

void dtiTractographyImageFilter::SetInputImage(ModelImageType *input)
{
    this->Superclass::SetInputImage(input);

    m_DTIInterpolator = DTIInterpolatorType::New();
    m_DTIInterpolator->SetInputImage(this->GetInputImage());
}

void dtiTractographyImageFilter::GetScalarValues(const VectorType &modelValue, double &mdValue, double &faValue)
{
    typedef vnl_matrix <double> MatrixType;

    itk::SymmetricEigenAnalysis <MatrixType,vnl_diag_matrix <double>,MatrixType> EigenAnalysis(3);
    MatrixType tmpMat(3,3);

    anima::GetTensorFromVectorRepresentation(modelValue,tmpMat);

    vnl_diag_matrix <double> eVals(3);
    EigenAnalysis.ComputeEigenValues(tmpMat,eVals);

    mdValue = 0;
    for (unsigned int i = 0;i < 3;++i)
    {
        eVals[i] = std::exp(eVals[i]);
        mdValue += eVals[i];
    }

    mdValue /= 3.0;

    double num = 0;
    double denom = 0;
    for (unsigned int i = 0;i < 3;++i)
    {
        num += (eVals[i] - mdValue) * (eVals[i] - mdValue);
        denom += eVals[i] * eVals[i];
    }

    faValue = std::sqrt(3.0 * num / (2.0 * denom));
}

bool dtiTractographyImageFilter::CheckModelCompatibility(VectorType &modelValue, itk::ThreadIdType threadId)
{
    double mdValue, faValue;
    this->GetScalarValues(modelValue, mdValue, faValue);

    if (mdValue > m_StopADCThreshold)
        return false;

    if (faValue < m_StopFAThreshold)
        return false;

    return true;
}

bool dtiTractographyImageFilter::CheckIndexInImageBounds(ContinuousIndexType &index)
{
    return m_DTIInterpolator->IsInsideBuffer(index);
}

dtiTractographyImageFilter::VectorType dtiTractographyImageFilter::GetModelValue(ContinuousIndexType &index, PointType &currentDirection, itk::ThreadIdType threadId)
{
    return m_DTIInterpolator->EvaluateAtContinuousIndex(index);
}

std::vector <dtiTractographyImageFilter::PointType>
dtiTractographyImageFilter::GetModelPrincipalDirections(VectorType &modelValue, bool is2d, itk::ThreadIdType threadId)
{
    typedef vnl_matrix <double> MatrixType;

    itk::SymmetricEigenAnalysis <MatrixType,vnl_diag_matrix <double>,MatrixType> EigenAnalysis(3);

    MatrixType tmpMat(3,3);
    vnl_diag_matrix <double> eVals(3);
    MatrixType eVec(3,3);

    anima::GetTensorFromVectorRepresentation(modelValue,tmpMat);

    EigenAnalysis.ComputeEigenValuesAndVectors(tmpMat,eVals,eVec);
    std::vector <PointType> resDir(1);

    for (unsigned int i = 0;i < 3;++i)
        resDir[0][i] = eVec(2,i);

    if (is2d)
    {
        resDir[0][2] = 0;
        anima::Normalize(resDir[0],resDir[0]);
    }

    return resDir;
}

dtiTractographyImageFilter::PointType
dtiTractographyImageFilter::GetNextDirection(PointType &previousDirection, VectorType &modelValue, bool is2d,
                                             itk::ThreadIdType threadId)
{
    PointType newDirection = this->GetModelPrincipalDirections(modelValue,is2d,threadId)[0];
    if (anima::ComputeScalarProduct(previousDirection, newDirection) < 0)
        anima::Revert(newDirection,newDirection);

    typedef vnl_matrix <double> MatrixType;
    MatrixType tmpMat(3,3);

    anima::GetTensorFromVectorRepresentation(modelValue,tmpMat,3,false);

    vnl_diag_matrix <double> eVals(3);
    itk::SymmetricEigenAnalysis <MatrixType,vnl_diag_matrix <double>,MatrixType> EigenAnalysis(3);
    MatrixType eVecs(3,3);
    EigenAnalysis.ComputeEigenValuesAndVectors(tmpMat,eVals, eVecs);

    double sumEigs = 0.0;
    for (unsigned int i = 0;i < 3;++i)
    {
        eVals[i] = std::exp(eVals[i]);
        sumEigs += eVals[i];
    }

    anima::RecomposeTensor(eVals,eVecs,tmpMat);

    // Compute advected direction as in Weinstein et al.
    PointType advectedDirection;

    for (unsigned int i = 0;i < 3;++i)
    {
        advectedDirection[i] = 0.0;
        for (unsigned int j = 0;j < 3;++j)
            advectedDirection[i] += tmpMat(i,j) * previousDirection[j];
    }

    anima::Normalize(advectedDirection,advectedDirection);
    if (anima::ComputeScalarProduct(previousDirection, advectedDirection) < 0)
        anima::Revert(advectedDirection,advectedDirection);

    double linearCoefficient = (eVals[2] - eVals[1]) / sumEigs;

    for (unsigned int i = 0;i < 3;++i)
        newDirection[i] = newDirection[i] * linearCoefficient + (1.0 - linearCoefficient) * ((1.0 - this->GetPunctureWeight()) * previousDirection[i] + this->GetPunctureWeight() * advectedDirection[i]);

    anima::Normalize(newDirection,newDirection);

    return newDirection;
}


void dtiTractographyImageFilter::ComputeAdditionalScalarMaps()
{
    vtkSmartPointer <vtkPolyData> outputPtr = this->GetOutput();

    unsigned int numPoints = outputPtr->GetPoints()->GetNumberOfPoints();
    vtkPoints *myPoints = outputPtr->GetPoints();

    vtkSmartPointer <vtkDoubleArray> faArray = vtkDoubleArray::New();
    faArray->SetNumberOfComponents(1);
    faArray->SetName("FA");

    vtkSmartPointer <vtkDoubleArray> adcArray = vtkDoubleArray::New();
    adcArray->SetNumberOfComponents(1);
    adcArray->SetName("ADC");

    PointType tmpPoint;
    DTIInterpolatorType::ContinuousIndexType tmpIndex;

    VectorType tensorValue(6);
    tensorValue.Fill(0.0);
    PointType currentDir;
    currentDir.Fill(0.0);

    anima::LogEuclideanTensorCalculator <double>::Pointer leCalculator = anima::LogEuclideanTensorCalculator <double>::New();

    for (unsigned int i = 0;i < numPoints;++i)
    {
        for (unsigned int j = 0;j < 3;++j)
            tmpPoint[j] = myPoints->GetPoint(i)[j];

        this->GetInputImage()->TransformPhysicalPointToContinuousIndex(tmpPoint,tmpIndex);
        if (m_DTIInterpolator->IsInsideBuffer(tmpIndex))
            tensorValue = this->GetModelValue(tmpIndex,currentDir,0);

        anima::GetTensorFromVectorRepresentation(tensorValue,tmpMat,3,false);
        leCalculator->GetTensorExponential(tmpMat,tmpMat);

        double adcValue = 0;
        for (unsigned int j = 0;j < 3;++j)
            adcValue += tmpMat(j,j);

        adcArray->InsertNextValue(adcValue / 3.0);

        double faValue = 0;
        double faValueDenom = 0;
        EigenAnalysis.ComputeEigenValues(tmpMat,eVals);
        for (unsigned int j = 0;j < 3;++j)
        {
            faValueDenom += eVals[j] * eVals[j];
            for (unsigned int k = j + 1;k < 3;++k)
                faValue += (eVals[j] - eVals[k]) * (eVals[j] - eVals[k]);
        }

        faValue = std::sqrt(faValue / (2.0 * faValueDenom));
        faArray->InsertNextValue(faValue);
    }

    outputPtr->GetPointData()->AddArray(faArray);
    outputPtr->GetPointData()->AddArray(adcArray);
}

} // end of namespace anima
