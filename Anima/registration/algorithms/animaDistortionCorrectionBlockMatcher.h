#pragma once
#include <animaBaseBlockMatcher.h>

namespace anima
{

template <typename TInputImageType>
class DistortionCorrectionBlockMatcher : public anima::BaseBlockMatcher <TInputImageType>
{
public:
    DistortionCorrectionBlockMatcher();
    virtual ~DistortionCorrectionBlockMatcher() {}

    // To do: Contains symmetric correlation for now. Normally should be deduced from the rest
    // (attraction mode or extra middle image)
    enum SimilarityDefinition
    {
        MeanSquares = 0,
        Correlation,
        SquaredCorrelation,
        SymmetricCorrelation
    };

    enum TransformDefinition
    {
        Direction = 0,
        DirectionScale,
        DirectionScaleSkew
    };

    typedef BaseBlockMatcher <TInputImageType> Superclass;
    typedef typename Superclass::InputImageType InputImageType;
    typedef typename Superclass::InputImagePointer InputImagePointer;
    typedef typename Superclass::PointType PointType;
    typedef typename Superclass::AgregatorType AgregatorType;
    typedef typename Superclass::MetricPointer MetricPointer;
    typedef typename Superclass::BaseInputTransformPointer BaseInputTransformPointer;
    typedef typename Superclass::OptimizerPointer OptimizerPointer;

    typename AgregatorType::TRANSFORM_TYPE GetAgregatorInputTransformType() ITK_OVERRIDE;
    void SetBlockTransformType(TransformDefinition val) {m_BlockTransformType = val;}
    TransformDefinition &GetBlockTransformType() {return m_BlockTransformType;}

    void SetSearchSkewRadius(double val) {m_SearchSkewRadius = val;}
    void SetSearchScaleRadius(double val) {m_SearchScaleRadius = val;}

    void SetTranslateMax(double val) {m_TranslateMax = val;}
    void SetSkewMax(double val) {m_SkewMax = val;}
    void SetScaleMax(double val) {m_ScaleMax = val;}

    void SetTransformDirection(unsigned int val) {m_TransformDirection = val;}

    //! Set extra middle image used as target
    void SetExtraMiddleImage(InputImageType *image) {m_ExtraMiddleImage = image;}
    void SetAttractorMode(bool mode) {m_AttractorMode = mode;}

    bool GetMaximizedMetric() ITK_OVERRIDE;
    void SetSimilarityType(SimilarityDefinition val) {m_SimilarityType = val;}

protected:
    virtual BaseInputTransformPointer GetNewBlockTransform(PointType &blockCenter) ITK_OVERRIDE;

    void InitializeBlocks() ITK_OVERRIDE;
    virtual MetricPointer SetupMetric() ITK_OVERRIDE;
    virtual double ComputeBlockWeight(double val, unsigned int block) ITK_OVERRIDE;

    virtual void BlockMatchingSetup(MetricPointer &metric, unsigned int block) ITK_OVERRIDE;
    virtual void TransformDependantOptimizerSetup(OptimizerPointer &optimizer) ITK_OVERRIDE;

private:
    SimilarityDefinition m_SimilarityType;
    TransformDefinition m_BlockTransformType;

    //! Boolean handling attractor mode switch
    bool m_AttractorMode;
    //! External reference image for symmetric correlation
    InputImagePointer m_ExtraMiddleImage;
    //! Artificial middle image if we are working in attractor mode
    InputImagePointer m_ArtificialMiddleImage;

    // Bobyqa radiuses
    double m_SearchSkewRadius;
    double m_SearchScaleRadius;

    //Bobyqa bounds parameters
    double m_TranslateMax;
    double m_SkewMax;
    double m_ScaleMax;

    unsigned int m_TransformDirection;
};

} // end namespace anima

#include "animaDistortionCorrectionBlockMatcher.hxx"
