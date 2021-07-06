#pragma once

#include <itkVectorImage.h>
#include <itkImage.h>

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <itkProcessObject.h>
#include <itkLinearInterpolateImageFunction.h>
#include <mutex>
#include <itkProgressReporter.h>

#include <vector>
#include <random>

namespace anima
{

template <class TInputModelImageType>
class BaseProbabilisticTractographyImageFilter : public itk::ProcessObject
{
public:
    /** SmartPointer typedef support  */
    typedef BaseProbabilisticTractographyImageFilter Self;
    typedef itk::ProcessObject Superclass;

    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    itkTypeMacro(BaseProbabilisticTractographyImageFilter,itk::ProcessObject)

    // Typdefs for scalar types for reading/writing images and for math operations
    typedef double ScalarType;

    // Typdef for input model image
    typedef TInputModelImageType InputModelImageType;
    typedef typename InputModelImageType::Pointer InputModelImagePointer;

    // Typedefs for B0 and noise images
    typedef itk::Image <ScalarType, 3> ScalarImageType;
    typedef typename ScalarImageType::Pointer ScalarImagePointer;
    typedef itk::LinearInterpolateImageFunction <ScalarImageType> ScalarInterpolatorType;
    typedef typename ScalarInterpolatorType::Pointer ScalarInterpolatorPointer;

    // Typedefs for input mask image
    typedef itk::Image <unsigned short, 3> MaskImageType;
    typedef MaskImageType::Pointer MaskImagePointer;
    typedef MaskImageType::PointType PointType;
    typedef MaskImageType::IndexType IndexType;

    // Typedefs for vectors and matrices
    typedef itk::Matrix <ScalarType,3,3> Matrix3DType;
    typedef itk::Vector <ScalarType,3> Vector3DType;
    typedef itk::VariableLengthVector <ScalarType> VectorType;
    typedef std::vector <ScalarType> ListType;
    typedef std::vector <Vector3DType> DirectionVectorType;

    // Typedefs for model images interpolator
    typedef itk::InterpolateImageFunction <InputModelImageType> InterpolatorType;
    typedef typename InterpolatorType::Pointer InterpolatorPointer;
    typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;

    // Typdefs for fibers
    typedef std::vector <PointType> FiberType;
    typedef std::vector <FiberType> FiberProcessVectorType;
    typedef std::vector <unsigned int> MembershipType;

    typedef struct {
        BaseProbabilisticTractographyImageFilter *trackerPtr;
        std::vector <FiberProcessVectorType> resultFibersFromThreads;
        std::vector <ListType> resultWeightsFromThreads;
    } trackerArguments;

    struct pair_comparator
    {
        bool operator() (const std::pair<unsigned int, double> & f, const std::pair<unsigned int, double> & s)
        { return (f.second < s.second); }
    };

    struct FiberWorkType
    {
        FiberProcessVectorType fiberParticles;
        MembershipType classMemberships;
        std::vector <MembershipType> reverseClassMemberships;
        MembershipType classSizes;
        ListType logParticleWeights, logNormalizedParticleWeights;
        ListType previousUpdateLogWeights;
        ListType logClassWeights;
        std::vector <unsigned int> fiberNumberOfPoints;
        std::vector <bool> stoppedParticles;
    };

    void AddGradientDirection(unsigned int i, Vector3DType &grad);
    Vector3DType &GetDiffusionGradient(unsigned int i) {return m_DiffusionGradients[i];}
    void SetBValuesList(ListType bValuesList) {m_BValuesList = bValuesList;}
    ScalarType GetBValueItem(unsigned int i) {return m_BValuesList[i];}
    unsigned int GetNumberOfGradientDirections() {return m_BValuesList.size();}

    virtual void SetInputModelImage(InputModelImageType *inImage) {m_InputModelImage = inImage;}
    InputModelImageType *GetInputModelImage() {return m_InputModelImage;}
    virtual InterpolatorType *GetModelInterpolator();

    itkSetObjectMacro(SeedMask,MaskImageType)
    itkSetObjectMacro(FilterMask,MaskImageType)
    itkSetObjectMacro(CutMask,MaskImageType)
    itkSetObjectMacro(ForbiddenMask,MaskImageType)

    itkSetObjectMacro(B0Image,ScalarImageType)
    itkSetObjectMacro(NoiseImage,ScalarImageType)

    itkSetMacro(NumberOfParticles,unsigned int)
    itkSetMacro(NumberOfFibersPerPixel,unsigned int)
    itkSetMacro(ResamplingThreshold,double)

    itkSetMacro(StepProgression,double)

    itkSetMacro(MinLengthFiber,double)
    itkSetMacro(MaxLengthFiber,double)

    itkSetMacro(FiberTrashThreshold,double)

    itkSetMacro(KappaOfPriorDistribution,double)
    itkGetMacro(KappaOfPriorDistribution,double)

    itkSetMacro(PositionDistanceFuseThreshold,double)
    itkSetMacro(KappaSplitThreshold,double)

    itkSetMacro(ClusterDistance,unsigned int)

    itkSetMacro(ComputeLocalColors,bool)
    itkSetMacro(MAPMergeFibers,bool)

    itkSetMacro(MinimalNumberOfParticlesPerClass,unsigned int)

    itkSetMacro(ModelDimension, unsigned int)
    itkGetMacro(ModelDimension, unsigned int)

    void Update() ITK_OVERRIDE;

    void createVTKOutput(FiberProcessVectorType &filteredFibers, ListType &filteredWeights);
    vtkPolyData *GetOutput() {return m_Output;}

protected:
    BaseProbabilisticTractographyImageFilter();
    virtual ~BaseProbabilisticTractographyImageFilter();

    //! Multithread util function
    static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION ThreadTracker(void *arg);

    //! Doing the thread work dispatch
    void ThreadTrack(unsigned int numThread, FiberProcessVectorType &resultFibers, ListType &resultWeights);

    //! Doing the real tracking by calling ComputeFiber and merging its results
    void ThreadedTrackComputer(unsigned int numThread, FiberProcessVectorType &resultFibers,
                               ListType &resultWeights, unsigned int startSeedIndex,
                               unsigned int endSeedIndex);

    //! This little guy is the one handling probabilistic tracking
    FiberProcessVectorType ComputeFiber(FiberType &fiber, InterpolatorPointer &modelInterpolator,
                                        unsigned int numThread, ListType &resultWeights);

    //! Generate seed points (can be re-implemented but this one has to be called)
    virtual void PrepareTractography();

    //! This ugly guy is the heart of multi-modal probabilistic tractography, making decisions on split and merges of particles
    unsigned int UpdateClassesMemberships(FiberWorkType &fiberData, DirectionVectorType &directions, std::mt19937 &random_generator);

    //! Performs the weight update part at each iteration
    void UpdateWeightsFromCurrentData(FiberWorkType &fiberComputationData, ListType &logWeightSums);

    //! Check if occasional resampling is needed. If yes, does it
    void CheckAndPerformOccasionalResampling(FiberWorkType &fiberComputationData, DirectionVectorType &previousDirections,
                                             DirectionVectorType &previousDirectionsCopy, unsigned int numThread);

    //! Initialize first direction from user input (model dependent, not implemented here)
    virtual void InitializeFirstIterationFromModel(VectorType &modelValue, unsigned int threadId,
                                                   DirectionVectorType &initialDirections);

    //! Make the particles move forward one step
    void ProgressParticles(FiberWorkType &fiberComputationData, InterpolatorPointer &modelInterpolator,
                           DirectionVectorType &previousDirections, unsigned int numThread);

    //! This guy takes the result of computefiber and merges the classes, each one becomes one fiber
    // Returns in outputMerged several fibers, as of now if there are active particles it returns only the merge of those, and returns true.
    // Otherwise, returns false and a merge per stopped fiber lengths
    bool MergeParticleClassFibers(FiberWorkType &fiberData, FiberProcessVectorType &outputMerged, unsigned int classNumber);

    //! Filter output fibers by ROIs and compute local colors
    FiberProcessVectorType FilterOutputFibers(FiberProcessVectorType &fibers, ListType &weights);

    //! Computes additional scalar maps that are model dependent to add to the output
    virtual void ComputeAdditionalScalarMaps() {}

    //////////
    // Pure virtual methods, to be defined in child classes

    //! Propose new direction for a particle, given the old direction, and a model (model dependent, not implemented here)
    virtual Vector3DType ProposeNewDirection(Vector3DType &oldDirection, VectorType &modelValue, std::mt19937 &random_generator,
                                             unsigned int threadId) = 0;

    //! Update particle weight based on an underlying model and the chosen direction (model dependent, not implemented here)
    virtual double ComputeLogWeightUpdate(double b0Value, double noiseValue, Vector3DType &previousDirection,
                                          Vector3DType &newDirection, VectorType &previousModelValue, VectorType &modelValue,
                                          unsigned int threadId) = 0;

    //! Estimate model from raw diffusion data (model dependent, not implemented here)
    virtual void ComputeModelValue(InterpolatorPointer &modelInterpolator, ContinuousIndexType &index, VectorType &modelValue) = 0;

    //! Check stopping criterions to stop a particle (model dependent, not implemented here)
    virtual bool CheckModelProperties(double estimatedB0Value, double estimatedNoiseValue, VectorType &modelValue, unsigned int threadId) = 0;

private:
    ITK_DISALLOW_COPY_AND_ASSIGN(BaseProbabilisticTractographyImageFilter);

    //Internal variable for model vector dimension, has to be set by child class !
    unsigned int m_ModelDimension;

    DirectionVectorType m_DiffusionGradients;
    ListType m_BValuesList;

    unsigned int m_NumberOfParticles;
    unsigned int m_NumberOfFibersPerPixel;
    unsigned int m_MinimalNumberOfParticlesPerClass;

    double m_StepProgression;

    double m_MinLengthFiber;
    double m_MaxLengthFiber;

    double m_FiberTrashThreshold;

    double m_ResamplingThreshold;

    double m_KappaOfPriorDistribution;

    InputModelImagePointer m_InputModelImage;

    MaskImagePointer m_SeedMask;
    MaskImagePointer m_FilterMask;
    MaskImagePointer m_CutMask;
    MaskImagePointer m_ForbiddenMask;

    ScalarImagePointer m_B0Image, m_NoiseImage;
    ScalarInterpolatorPointer m_B0Interpolator, m_NoiseInterpolator;

    std::vector <std::mt19937> m_Generators;

    FiberProcessVectorType m_PointsToProcess;
    MembershipType m_FilteringValues;

    // Multimodal splitting and merging thresholds
    double m_PositionDistanceFuseThreshold;
    double m_KappaSplitThreshold;

    unsigned int m_ClusterDistance;

    bool m_MAPMergeFibers;
    bool m_ComputeLocalColors;

    vtkSmartPointer<vtkPolyData> m_Output;

    std::mutex m_LockHighestProcessedSeed;
    int m_HighestProcessedSeed;
    itk::ProgressReporter *m_ProgressReport;
};

}//end of namesapce

#include "animaBaseProbabilisticTractographyImageFilter.hxx"
