#include <animaMCMWeightedAverager.h>

namespace anima
{

const vnl_matrix<double> MCMWeightedAverager::m_ReferenceSigma = vnl_diag_matrix<double>(3,3.0e-3);

MCMWeightedAverager::MCMWeightedAverager()
{
    m_UpToDate = false;
    m_TensorInterpolationMethod = 0;
    m_NumberOfOutputDirectionalCompartments = 3;
    m_InternalEigenAnalyzer.SetDimension(3);
    m_InternalEigenAnalyzer.SetOrder(3);

    m_InternalSpectralCluster.SetMaxIterations(200);
    m_InternalSpectralCluster.SetCMeansAverageType(SpectralClusterType::CMeansFilterType::Euclidean);
    m_InternalSpectralCluster.SetVerbose(false);

    m_leCalculator = LECalculatorType::New();
}

void MCMWeightedAverager::SetOutputModel(MCMType *model)
{
    m_OutputModel = model->Clone();
    this->ResetNumberOfOutputDirectionalCompartments();
}

void MCMWeightedAverager::SetTensorInterpolationMethod(unsigned int method)
{
    if (m_TensorInterpolationMethod == method)
        return;

    m_TensorInterpolationMethod = method;
    m_UpToDate = false;
}

void MCMWeightedAverager::SetNumberOfOutputDirectionalCompartments(unsigned int val)
{
    m_NumberOfOutputDirectionalCompartments = val;
    m_UpToDate = false;
}

void MCMWeightedAverager::ResetNumberOfOutputDirectionalCompartments()
{
    m_NumberOfOutputDirectionalCompartments = m_OutputModel->GetNumberOfCompartments() - m_OutputModel->GetNumberOfIsotropicCompartments();
    m_UpToDate = false;
}

MCMWeightedAverager::MCMPointer &MCMWeightedAverager::GetOutputModel()
{
    if (!m_UpToDate)
        this->Update();

    return m_OutputModel;
}

MCMWeightedAverager::MCMPointer &MCMWeightedAverager::GetUntouchedOutputModel()
{
    return m_OutputModel;
}

unsigned int MCMWeightedAverager::GetOutputModelSize()
{
    if (!m_OutputModel)
        itkExceptionMacro("Output model not initialized")

    return m_OutputModel->GetSize();
}

void MCMWeightedAverager::Update()
{
    if (m_UpToDate)
        return;

    if (m_InputModels.size() != m_InputWeights.size())
        itkExceptionMacro("Not the same number of weights and input models");

    unsigned int numInputs = m_InputModels.size();
    // First make sure weights sum up to 1
    double sumWeights = 0;
    for (unsigned int i = 0;i < numInputs;++i)
        sumWeights += m_InputWeights[i];

    for (unsigned int i = 0;i < numInputs;++i)
        m_InputWeights[i] /= sumWeights;

    // Perform isotropic models average
    unsigned int numIsoCompartments = m_InputModels[0]->GetNumberOfIsotropicCompartments();
    m_InternalOutputWeights.resize(m_OutputModel->GetNumberOfCompartments());
    std::fill(m_InternalOutputWeights.begin(),m_InternalOutputWeights.end(),0.0);

    vnl_matrix <double> meanMatrix(3,3), workIsoMatrix(3,3);
    for (unsigned int i = 0;i < numIsoCompartments;++i)
    {
        if (m_TensorInterpolationMethod == 0)
        {
            double outputLogDiffusivity = 0;
            double outputRadius = 0.0;
            double sumWeights = 0;
            for (unsigned int j = 0;j < numInputs;++j)
            {
                if (m_InputWeights[j] <= 0)
                    continue;

                double tmpWeight = m_InputModels[j]->GetCompartmentWeight(i);
                if (tmpWeight <= 0)
                    continue;

                m_InternalOutputWeights[i] += m_InputWeights[j] * tmpWeight;
                outputLogDiffusivity += m_InputWeights[j] * std::log(m_InputModels[j]->GetCompartment(i)->GetAxialDiffusivity());
                outputRadius += m_InputWeights[j] * m_InputModels[j]->GetCompartment(i)->GetTissueRadius();
                sumWeights += m_InputWeights[j];
            }

            if (sumWeights > 0)
            {
                m_OutputModel->GetCompartment(i)->SetAxialDiffusivity(std::exp(outputLogDiffusivity / sumWeights));
                m_OutputModel->GetCompartment(i)->SetTissueRadius(outputRadius / sumWeights);
            }
        }
        else
        {
            // Bayes interpolation of isotropic compartments
            double sumWeights = 0;
            for (unsigned int j = 0;j < numInputs;++j)
            {
                if ((m_InputWeights[j] > 0) && (m_InputModels[j]->GetCompartmentWeight(i) > 0))
                    sumWeights += 1.0;
            }

            meanMatrix = m_ReferenceSigma;
            for (unsigned int j = 0;j < numInputs;++j)
            {
                if (m_InputWeights[j] <= 0)
                    continue;

                double tmpWeight = m_InputModels[j]->GetCompartmentWeight(i);
                if (tmpWeight <= 0)
                    continue;

                m_InternalOutputWeights[i] += m_InputWeights[j] * tmpWeight;

                workIsoMatrix = m_InputModels[j]->GetCompartment(i)->GetDiffusionTensor().GetVnlMatrix().as_matrix();
                anima::GetTensorPower(workIsoMatrix,workIsoMatrix,-1.0);
                workIsoMatrix = anima::GetZeroMeanGaussianScalarMultiplication(workIsoMatrix,1.0 / sumWeights,m_ReferenceSigma);
                meanMatrix = anima::GetZeroMeanGaussianAddition(meanMatrix,workIsoMatrix,m_ReferenceSigma);
            }

            if (sumWeights > 0)
                m_OutputModel->GetCompartment(i)->SetAxialDiffusivity(1.0 / meanMatrix(0,0));
        }
    }

    unsigned int maxNumOutputCompartments = m_OutputModel->GetNumberOfCompartments() - m_OutputModel->GetNumberOfIsotropicCompartments();
    unsigned int numOutputCompartments = m_NumberOfOutputDirectionalCompartments;
    if (numOutputCompartments > maxNumOutputCompartments)
        numOutputCompartments = maxNumOutputCompartments;

    m_WorkCompartmentsVector.clear();
    m_WorkCompartmentWeights.clear();

    for (unsigned int i = 0;i < numInputs;++i)
    {
        if (m_InputWeights[i] <= 0)
            continue;

        for (unsigned int j = numIsoCompartments;j < m_InputModels[i]->GetNumberOfCompartments();++j)
        {
            double tmpWeight = m_InputModels[i]->GetCompartmentWeight(j);
            if (tmpWeight <= 0)
                continue;

            m_WorkCompartmentsVector.push_back(m_InputModels[i]->GetCompartment(j));
            m_WorkCompartmentWeights.push_back(m_InputWeights[i] * tmpWeight);
        }
    }

    unsigned int numInputCompartments = m_WorkCompartmentsVector.size();

    if (numInputCompartments <= numOutputCompartments)
    {
        for (unsigned int i = 0;i < numInputCompartments;++i)
        {
            m_OutputModel->GetCompartment(i + numIsoCompartments)->SetCompartmentVector(m_WorkCompartmentsVector[i]->GetCompartmentVector());
            m_InternalOutputWeights[i + numIsoCompartments] = m_WorkCompartmentWeights[i];
        }

        m_OutputModel->SetCompartmentWeights(m_InternalOutputWeights);

        m_UpToDate = true;
        return;
    }

    bool tensorCompatibility = m_WorkCompartmentsVector[0]->GetTensorCompatible();
    if (tensorCompatibility)
    {
        if (m_TensorInterpolationMethod == 0)
            this->ComputeTensorLEDistanceMatrix();
        else
            this->ComputeTensorBayesDistanceMatrix();
    }
    else
        this->ComputeNonTensorDistanceMatrix();

    m_InternalSpectralCluster.SetNbClass(numOutputCompartments);
    m_InternalSpectralCluster.SetInputData(m_InternalDistanceMatrix);
    m_InternalSpectralCluster.SetDataWeights(m_WorkCompartmentWeights);
    m_InternalSpectralCluster.InitializeSigmaFromDistances();

    m_InternalSpectralCluster.Update();

    m_InternalSpectralMemberships.resize(numInputCompartments);
    for (unsigned int i = 0;i < numInputCompartments;++i)
        m_InternalSpectralMemberships[i] = m_InternalSpectralCluster.GetClassesMembership(i);

    if (tensorCompatibility)
    {
        if (m_TensorInterpolationMethod == 0)
            this->ComputeOutputLETensorModel();
        else
            this->ComputeOutputBayesTensorModel();
    }
    else
        this->ComputeOutputNonTensorModel();

    m_OutputModel->SetCompartmentWeights(m_InternalOutputWeights);

    m_UpToDate = true;
}

void MCMWeightedAverager::ComputeTensorLEDistanceMatrix()
{
    unsigned int numCompartments = m_WorkCompartmentsVector.size();
    m_InternalLogTensors.resize(numCompartments);

    for (unsigned int i = 0;i < numCompartments;++i)
    {
        m_InternalWorkMatrix = m_WorkCompartmentsVector[i]->GetDiffusionTensor().GetVnlMatrix().as_matrix();
        m_leCalculator->GetTensorLogarithm(m_InternalWorkMatrix,m_InternalWorkMatrix);
        anima::GetVectorRepresentation(m_InternalWorkMatrix,m_InternalLogTensors[i],6,true);
    }

    m_InternalDistanceMatrix.set_size(numCompartments,numCompartments);
    m_InternalDistanceMatrix.fill(0);

    for (unsigned int i = 0;i < numCompartments;++i)
        for (unsigned int j = i+1;j < numCompartments;++j)
        {
            double distValue = 0;
            for (unsigned int k = 0;k < 6;++k)
                distValue += (m_InternalLogTensors[i][k] - m_InternalLogTensors[j][k]) * (m_InternalLogTensors[i][k] - m_InternalLogTensors[j][k]);

            m_InternalDistanceMatrix(i,j) = distValue;
            m_InternalDistanceMatrix(j,i) = m_InternalDistanceMatrix(i,j);
        }
}

void MCMWeightedAverager::ComputeTensorBayesDistanceMatrix()
{
    unsigned int numCompartments = m_WorkCompartmentsVector.size();
    m_InternalInvertedTensors.resize(numCompartments);

    vnl_matrix <double> workMatrix(3,3);
    for (unsigned int i = 0;i < numCompartments;++i)
    {
        workMatrix = m_WorkCompartmentsVector[i]->GetDiffusionTensor().GetVnlMatrix().as_matrix();
        anima::GetTensorPower(workMatrix,m_InternalInvertedTensors[i],-1.0);
    }

    m_InternalDistanceMatrix.set_size(numCompartments,numCompartments);
    m_InternalDistanceMatrix.fill(0);

    for (unsigned int i = 0;i < numCompartments;++i)
    {
        for (unsigned int j = i+1;j < numCompartments;++j)
        {
            double distValue = anima::GetZeroMeanGaussianSquaredDistance(m_InternalInvertedTensors[i],
                                                                         m_InternalInvertedTensors[j],
                                                                         3.0e-3);
            m_InternalDistanceMatrix(i,j) = distValue;
            m_InternalDistanceMatrix(j,i) = m_InternalDistanceMatrix(i,j);
        }
    }
}

void MCMWeightedAverager::ComputeNonTensorDistanceMatrix()
{
    itkExceptionMacro("No non-tensor distance matrix implemented in public version")
}

void MCMWeightedAverager::ComputeOutputLETensorModel()
{
    unsigned int numCompartments = m_WorkCompartmentsVector.size();
    unsigned int numIsoCompartments = m_OutputModel->GetNumberOfIsotropicCompartments();
    unsigned int numberOfOutputCompartments = m_InternalSpectralMemberships[0].size();

    m_InternalWorkMatrix.set_size(3,3);
    m_InternalWorkEigenVectors.set_size(3,3);
    m_InternalWorkEigenValues.set_size(3);

    m_InternalOutputVector.SetSize(6);
    m_InternalWorkEigenValuesInputSticks.set_size(3);

    anima::DiffusionModelCompartmentType anisoCompartmentType = m_OutputModel->GetCompartment(numIsoCompartments)->GetCompartmentType();

    for (unsigned int i = 0;i < numberOfOutputCompartments;++i)
    {
        m_InternalOutputVector.Fill(0);
        double totalWeights = 0;
        for (unsigned int j = 0;j < numCompartments;++j)
        {
            double weight = m_WorkCompartmentWeights[j] * m_InternalSpectralMemberships[j][i];
            if (weight == 0)
                continue;

            // Stick is with a fixed radial diffusivity, get it for later setting it back
            if ((anisoCompartmentType == anima::Stick) && (totalWeights == 0.0))
            {
                anima::GetTensorFromVectorRepresentation(m_InternalLogTensors[j],m_InternalWorkMatrix,3,true);
                m_InternalEigenAnalyzer.ComputeEigenValues(m_InternalWorkMatrix, m_InternalWorkEigenValuesInputSticks);
            }

            totalWeights += weight;
            m_InternalOutputVector += m_InternalLogTensors[j] * weight;
        }

        if (totalWeights > 0.0)
        {
            m_InternalOutputVector /= totalWeights;

            m_InternalOutputWeights[i+numIsoCompartments] = totalWeights;
            anima::GetTensorFromVectorRepresentation(m_InternalOutputVector,m_InternalWorkMatrix,3,true);

            if (anisoCompartmentType != anima::Tensor)
                m_InternalEigenAnalyzer.ComputeEigenValuesAndVectors(m_InternalWorkMatrix, m_InternalWorkEigenValues, m_InternalWorkEigenVectors);

            if (anisoCompartmentType == anima::Stick)
            {
                // Replace smaller eigen values by stick default value
                m_InternalWorkEigenValues[0] = m_InternalWorkEigenValuesInputSticks[0];
                m_InternalWorkEigenValues[1] = m_InternalWorkEigenValuesInputSticks[0];

                anima::RecomposeTensor(m_InternalWorkEigenValues, m_InternalWorkEigenVectors, m_InternalWorkMatrix);
            }
            else if (anisoCompartmentType == anima::Zeppelin)
            {
                // Force radial diffusivity as the sum of the two smallest ones
                double logEigenValue = (m_InternalWorkEigenValues[0] + m_InternalWorkEigenValues[1]) / 2.0;
                m_InternalWorkEigenValues[0] = logEigenValue;
                m_InternalWorkEigenValues[1] = logEigenValue;

                anima::RecomposeTensor(m_InternalWorkEigenValues, m_InternalWorkEigenVectors, m_InternalWorkMatrix);
            }

            m_leCalculator->GetTensorExponential(m_InternalWorkMatrix,m_InternalWorkMatrix);
            anima::GetVectorRepresentation(m_InternalWorkMatrix,m_InternalOutputVector);
        }

        anima::BaseCompartment *workCompartment = m_OutputModel->GetCompartment(i+numIsoCompartments);
        workCompartment->SetCompartmentVector(m_InternalOutputVector);
    }
}

void MCMWeightedAverager::ComputeOutputBayesTensorModel()
{
    unsigned int numCompartments = m_WorkCompartmentsVector.size();
    unsigned int numIsoCompartments = m_OutputModel->GetNumberOfIsotropicCompartments();
    unsigned int numberOfOutputCompartments = m_InternalSpectralMemberships[0].size();

    itk::VariableLengthVector <double> outputVector(6);
    vnl_matrix <double> workMatrix(3,3), meanMatrix(3,3);

    for (unsigned int i = 0;i < numberOfOutputCompartments;++i)
    {
        outputVector.Fill(0);
        meanMatrix = m_ReferenceSigma;

        double sumWeights = 0;
        for (unsigned int j = 0;j < numCompartments;++j)
        {
            if (m_WorkCompartmentWeights[j] * m_InternalSpectralMemberships[j][i] > 0)
                sumWeights += m_InternalSpectralMemberships[j][i];
        }

        double totalWeights = 0;
        for (unsigned int j = 0;j < numCompartments;++j)
        {
            double weight = m_WorkCompartmentWeights[j] * m_InternalSpectralMemberships[j][i];
            if (weight == 0)
                continue;

            totalWeights += weight;

            workMatrix = anima::GetZeroMeanGaussianScalarMultiplication(m_InternalInvertedTensors[j],m_InternalSpectralMemberships[j][i] / sumWeights,m_ReferenceSigma);
            meanMatrix = anima::GetZeroMeanGaussianAddition(meanMatrix,workMatrix,m_ReferenceSigma);
        }

        anima::GetTensorPower(meanMatrix,meanMatrix,-1.0);
        anima::GetVectorRepresentation(meanMatrix,outputVector,6,false);

        m_InternalOutputWeights[i+numIsoCompartments] = totalWeights;

        anima::BaseCompartment *workCompartment = m_OutputModel->GetCompartment(i+numIsoCompartments);
        workCompartment->SetCompartmentVector(outputVector);
    }
}

void MCMWeightedAverager::ComputeOutputNonTensorModel()
{
    itkExceptionMacro("No non-tensor model computation implemented in public version")
}

} // end namespace anima
