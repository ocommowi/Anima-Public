#pragma once
#include <AnimaFibersAnalysisExport.h>

#include <animaSingleFiberUnbalancedOptimalTransport.h>

#include <itkProcessObject.h>
#include <vnl/vnl_matrix.h>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkDoubleArray.h>

#include <vector>

namespace anima
{

class ANIMAFIBERSANALYSIS_EXPORT FiberBundleUnbalancedOptimalTransport : public itk::ProcessObject
{
public:
    /** SmartPointer typedef support  */
    typedef FiberBundleUnbalancedOptimalTransport Self;
    typedef itk::ProcessObject Superclass;

    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    typedef anima::SingleFiberUnbalancedOptimalTransport FiberDistanceComputerType;
    typedef FiberDistanceComputerType::Pointer FiberDistanceComputerPointer;

    itkNewMacro(Self)

    itkTypeMacro(FiberBundleUnbalancedOptimalTransport,itk::ProcessObject)

    virtual void Update() ITK_OVERRIDE;

    void SetFirstDataset(vtkPolyData *data) {m_FirstDataset = data;}
    vtkPolyData *GetFirstDataset() {return m_FirstDataset;}
    void SetSecondDataset(vtkPolyData *data) {m_SecondDataset = data;}
    vtkPolyData *GetSecondDataset() {return m_SecondDataset;}

    itkSetMacro(MemorySizeLimit, double)
    itkSetMacro(RhoValue, double)
    itkSetMacro(EpsilonValue, double)
    itkSetMacro(RelativeStopCriterion, double)
    itkSetMacro(AlphaValue, double)
    itkSetMacro(KValue, double)
    itkSetMacro(Verbose, bool)

    itkGetMacro(WassersteinSquaredDistance, double)

protected:
    FiberBundleUnbalancedOptimalTransport();
    virtual ~FiberBundleUnbalancedOptimalTransport() {}

    void PrepareInputFibersData();
    void PrecomputeDistanceMatrix();

    typedef struct {
        FiberBundleUnbalancedOptimalTransport *uotPtr;
    } ThreadArguments;

    static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION ThreadPrepare(void *arg);
    void ComputeExtraDataOnCell(unsigned int cellIndex, vtkGenericCell *cellData, unsigned int dataIndex);

    static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION ThreadPrecomputeDistanceMatrix(void *arg);
    void PrecomputeDistancesOnRange(unsigned int startIndex, unsigned int endIndex, unsigned int threadId);
    double ComputeDistance(unsigned int firstIndex, unsigned int secondIndex, unsigned int threadId);

    static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION ThreadVectorUpdate(void *arg);
    void ComputeVectorUpdateOnRange(unsigned int startIndex, unsigned int endIndex, unsigned int threadId);

    void ComputeWassersteinDistanceFomData();

    itkGetMacro(UpdateUVector, bool)

private:
    ITK_DISALLOW_COPY_AND_ASSIGN(FiberBundleUnbalancedOptimalTransport);

    //! The tested datasets. Each fiber has a 1.0 weight
    vtkSmartPointer <vtkPolyData> m_FirstDataset, m_SecondDataset;

    vtkSmartPointer <vtkDoubleArray> m_SegmentLengthsFirstDataset, m_SegmentLengthsSecondDataset;
    vtkSmartPointer <vtkDoubleArray> m_SegmentTangentsFirstDataset, m_SegmentTangentsSecondDataset;

    vnl_matrix <double> m_DistanceMatrix;
    double m_MemorySizeLimit;

    double m_WassersteinSquaredDistance;
    double m_AlphaValue, m_KValue;
    bool m_Verbose;

    // Sinkhorn variables
    std::vector <double> m_UVector, m_VVector;
    std::vector <double> m_OldUVector, m_OldVVector;
    double m_RhoValue, m_EpsilonValue;
    double m_RelativeStopCriterion;
    bool m_UpdateUVector;

    std::vector <FiberDistanceComputerPointer> m_DistanceComputers;
};

} // end namespace anima
