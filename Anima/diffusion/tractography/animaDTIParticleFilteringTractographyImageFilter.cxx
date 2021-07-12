#include "animaDTIParticleFilteringTractographyImageFilter.h"
#include <cmath>

#include <animaVectorOperations.h>
#include <animaDistributionSampling.h>
#include <animaVMFDistribution.h>
#include <animaWatsonDistribution.h>
#include <animaLogarithmFunctions.h>
#include <animaBaseTensorTools.h>

#include <itkSymmetricEigenAnalysis.h>

#include <vtkPointData.h>
#include <vtkDoubleArray.h>

namespace anima
{

DTIParticleFilteringTractographyImageFilter::DTIParticleFilteringTractographyImageFilter()
    : BaseParticleFilteringTractographyImageFilter()
{
    m_FAThreshold = 0.2;
    
    this->SetModelDimension(6);
}

DTIParticleFilteringTractographyImageFilter::~DTIParticleFilteringTractographyImageFilter()
{
}

DTIParticleFilteringTractographyImageFilter::Vector3DType
DTIParticleFilteringTractographyImageFilter::ProposeNewDirection(Vector3DType &oldDirection, VectorType &modelValue,
                                                                 std::mt19937 &random_generator, unsigned int threadId)
{
    Vector3DType meanVec(0.0), resVec(0.0);
    vnl_matrix <double> tensor(3,3);
    anima::GetTensorFromVectorRepresentation(modelValue, tensor);
    anima::SampleFromMultivariateGaussianDistribution(meanVec,tensor,resVec,random_generator);

    resVec.Normalize();
    bool is2d = (this->GetInputModelImage()->GetLargestPossibleRegion().GetSize()[2] <= 1);
    if (is2d)
    {
        resVec[InputModelImageType::ImageDimension - 1] = 0;
        resVec.Normalize();
    }

    if (anima::ComputeScalarProduct(oldDirection, resVec) < 0)
        resVec *= -1;

    return resVec;
}

bool DTIParticleFilteringTractographyImageFilter::CheckModelProperties(double estimatedB0Value, double estimatedNoiseValue, VectorType &modelValue, unsigned int threadId)
{
    if (estimatedB0Value < 50.0)
        return false;
    
    bool isTensorNull = true;
    for (unsigned int j = 0;j < this->GetModelDimension();++j)
    {
        if (modelValue[j] != 0)
        {
            isTensorNull = false;
            break;
        }
    }
    
    if (isTensorNull)
        return false;

    double fractionalAnisotropy = this->GetFractionalAnisotropy(modelValue);
    if (fractionalAnisotropy < m_FAThreshold)
        return false;
    
    return true;
}

double DTIParticleFilteringTractographyImageFilter::ComputeLogWeightUpdate(double b0Value, double noiseValue, Vector3DType &previousDirection,
                                                                           Vector3DType &newDirection, VectorType &previousModelValue, VectorType &modelValue,
                                                                           unsigned int threadId)
{
    bool is2d = this->GetInputModelImage()->GetLargestPossibleRegion().GetSize()[2] <= 1;

    // Prior is a Watson PDF
    double log_prior = anima::safe_log(anima::EvaluateWatsonPDF(newDirection, previousDirection, this->GetKappaOfPriorDistribution()));

    // Proposal is the probability to get the sampled new direction from the previous model:
    // here line integral of the tensor along new direction
    double log_proposal = 0.0;
    vnl_matrix <double> tensor(3,3);
    anima::GetTensorFromVectorRepresentation(previousModelValue, tensor);
    itk::SymmetricEigenAnalysis < vnl_matrix <double>, vnl_diag_matrix <double> , vnl_matrix <double> > EigenAnalysis(InputModelImageType::ImageDimension);

    vnl_matrix <double> eigVecs(3,3);
    vnl_diag_matrix <double> eigVals(3);

    EigenAnalysis.ComputeEigenValuesAndVectors(tensor,eigVals,eigVecs);

    double detPreviousTensor = 1.0;
    for (unsigned int i = 0;i < 3;++i)
    {
        detPreviousTensor *= eigVals[i];
        eigVals[i] = 1.0 / eigVals[i];
    }

    anima::RecomposeTensor(eigVals,eigVecs,tensor);
    double tensorFactor = 0.0;
    for (unsigned int i = 0;i < 3;++i)
    {
        for (unsigned int j = i + 1;j < 3;++j)
            tensorFactor += 2.0 * newDirection[i] * newDirection[j] * tensor(i,j);

        tensorFactor += newDirection[i] * newDirection[i] * tensor(i,i);
    }

    tensorFactor *= 0.5;
    // TO DO : check formula and correct where needed
    log_proposal = - 3 * std::log(2 * M_PI) - std::log(detPreviousTensor) + std::log(M_PI) - std::log(tensorFactor);
    log_proposal /= 2.0;

    // Log-likelihood: simulate from model at arrival and from current direction viewed as generic zeppelin tensor
    double log_likelihood = 0;

    unsigned int numGradientImages = this->GetNumberOfGradientDirections();
    Vector3DType gradient;
    anima::GetTensorFromVectorRepresentation(modelValue, tensor);
    vnl_matrix <double> zeppelinDirection(3,3);
    zeppelinDirection.fill(0.0);

    for (unsigned int i = 0;i < 3;++i)
    {
        zeppelinDirection(i,i) = 1.5e-4 + (1.71e-3 - 1.5e-4) * newDirection[i] * newDirection[i];
        for (unsigned int j = i + 1;j < 3;++j)
        {
            zeppelinDirection(i,j) = (1.71e-3 - 1.5e-4) * newDirection[i] * newDirection[j];
            zeppelinDirection(j,i) = zeppelinDirection(i,j);
        }
    }

    for (unsigned int i = 0;i < numGradientImages;++i)
    {
        double internalProductModel = 0.0;
        double internalProductDirection = 0.0;
        double bvalue = this->GetBValueItem(i);
        gradient = this->GetDiffusionGradient(i);

        for (unsigned int j = 0;j < 3;++j)
        {
            internalProductModel += gradient[j] * gradient[j] * tensor(j,j);
            internalProductDirection += gradient[j] * gradient[j] * zeppelinDirection(j,j);

            for (unsigned int k = j + 1;k < 3;++k)
            {
                internalProductModel += 2.0 * gradient[j] * gradient[k] * tensor(j,k);
                internalProductDirection += 2.0 * gradient[j] * gradient[k] * zeppelinDirection(j,k);
            }
        }

        internalProductModel *= bvalue;
        internalProductDirection *= bvalue;

        double diffValue = b0Value * (std::exp(- internalProductModel) - std::exp(- internalProductDirection));
        double logNoiseGaussValue = 0.5 * (std::log(noiseValue) + std::log(2.0 * M_PI) + diffValue * diffValue / noiseValue);

        log_likelihood -= logNoiseGaussValue;
    }
    
    double resVal = log_prior + log_likelihood - log_proposal;
    return resVal;
}

void DTIParticleFilteringTractographyImageFilter::ComputeModelValue(InterpolatorPointer &modelInterpolator, ContinuousIndexType &index,
                                                                    VectorType &modelValue)
{
    modelValue.SetSize(this->GetModelDimension());
    modelValue.Fill(0.0);
    
    if (modelInterpolator->IsInsideBuffer(index))
        modelValue = modelInterpolator->EvaluateAtContinuousIndex(index);

    using LECalculatorType = anima::LogEuclideanTensorCalculator <double>;
    using LECalculatorPointer = LECalculatorType::Pointer;

    LECalculatorPointer leCalculator = LECalculatorType::New();

    vnl_matrix <double> tmpTensor(3,3);
    anima::GetTensorFromVectorRepresentation(modelValue,tmpTensor);
    leCalculator->GetTensorExponential(tmpTensor,tmpTensor);
    anima::GetVectorRepresentation(tmpTensor,modelValue);
}

double DTIParticleFilteringTractographyImageFilter::GetFractionalAnisotropy(VectorType &modelValue)
{
    itk::SymmetricEigenAnalysis <Matrix3DType,Vector3DType,Matrix3DType> EigenAnalysis(InputModelImageType::ImageDimension);
    EigenAnalysis.SetOrderEigenValues(true);
    
    Matrix3DType dtiTensor;
    Vector3DType eigVals;
    
    unsigned int pos = 0;
    for (unsigned int i = 0;i < 3;++i)
        for (unsigned int j = 0;j <= i;++j)
        {
            dtiTensor(i,j) = modelValue[pos];
            if (j != i)
                dtiTensor(j,i) = dtiTensor(i,j);
            ++pos;
        }
    
    EigenAnalysis.ComputeEigenValues(dtiTensor,eigVals);
    
    double meanLambda = 0;
    for (unsigned int i = 0;i < 3;++i)
        meanLambda += eigVals[i];

    meanLambda /= 3.0;

    double num = 0;
    double denom = 0;
    for (unsigned int i = 0;i < 3;++i)
    {
        num += (eigVals[i] - meanLambda) * (eigVals[i] - meanLambda);
        denom += eigVals[i] * eigVals[i];
    }
    
    // FA
    return std::sqrt(3.0 * num / (2.0 * denom));
}

void DTIParticleFilteringTractographyImageFilter::ComputeAdditionalScalarMaps()
{
    vtkSmartPointer <vtkPolyData> outputPtr = this->GetOutput();

    InterpolatorPointer modelInterpolator = this->GetModelInterpolator();

    unsigned int numPoints = outputPtr->GetPoints()->GetNumberOfPoints();
    vtkPoints *myPoints = outputPtr->GetPoints();

    vtkSmartPointer <vtkDoubleArray> faArray = vtkDoubleArray::New();
    faArray->SetNumberOfComponents(1);
    faArray->SetName("FA");

    vtkSmartPointer <vtkDoubleArray> adcArray = vtkDoubleArray::New();
    adcArray->SetNumberOfComponents(1);
    adcArray->SetName("ADC");

    PointType tmpPoint;
    Superclass::InterpolatorType::ContinuousIndexType tmpIndex;

    typedef vnl_matrix <double> MatrixType;
    MatrixType tmpMat(3,3);
    vnl_diag_matrix <double> eVals(3);
    itk::SymmetricEigenAnalysis <MatrixType,vnl_diag_matrix <double>,MatrixType> EigenAnalysis(3);
    VectorType tensorValue(6);

    for (unsigned int i = 0;i < numPoints;++i)
    {
        for (unsigned int j = 0;j < 3;++j)
            tmpPoint[j] = myPoints->GetPoint(i)[j];

        this->GetInputModelImage()->TransformPhysicalPointToContinuousIndex(tmpPoint,tmpIndex);
        tensorValue.Fill(0.0);
        if (modelInterpolator->IsInsideBuffer(tmpIndex))
            this->ComputeModelValue(modelInterpolator,tmpIndex,tensorValue);

        anima::GetTensorFromVectorRepresentation(tensorValue,tmpMat,3,false);

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
