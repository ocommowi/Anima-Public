#pragma once
#include <cmath>

namespace anima
{

// The following units are in place to keep things right: distances are in millimeters, time in seconds, magnetic field in Tesla

//! Gyromagnetic ratio (in rad/s/T), from Nelson, J., Nuclear Magnetic Resonance Spectroscopy, Prentice Hall, Londres, 2003
const double DiffusionGyromagneticRatio = 267.513e6;

//! Given gyromagnetic ratio in rad/s/T, gradient strength in T/mm and deltas in s, computes b-value in s/mm^2
inline double GetBValueFromAcquisitionParameters(double smallDelta, double bigDelta, double gradientStrength)
{
    double alpha = DiffusionGyromagneticRatio * smallDelta * gradientStrength;
    return alpha * alpha * (bigDelta - smallDelta / 3.0);
}

//! Given b-value in s/mm^2 and deltas in s, computes gradient strength in T/mm
inline double GetGradientStrengthFromBValue(double bValue, double smallDelta, double bigDelta)
{
    double alpha = std::sqrt(bValue / (bigDelta - smallDelta / 3.0));
    return alpha / (DiffusionGyromagneticRatio * smallDelta);
}

//! Default small delta value (classical values)
const double DiffusionSmallDelta = 10.0e-3;

//! Default big delta value (classical values)
const double DiffusionBigDelta = 40.0e-3;

//! Default zero lower bound (in case we want something else than zero one day)
const double MCMZeroLowerBound = 0.0;

//! Epsilon value in case we do not want to have parameters reaching their true bounds (used for now in DDI)
const double MCMEpsilon = 1.0e-2;

//! Compartment fractions upper bound (B0 times weight)
const double MCMCompartmentsFractionUpperBound = 1.0e7;

//! Beta distribution prior alpha shape value
const double MCMPriorAlpha = 20.0;

//! Beta distribution prior beta shape value
const double MCMPriorBeta = 8.0;

//! Diffusivity lower bound for estimation
const double MCMDiffusivityLowerBound = 1.0e-6;

//! Diffusivity upper bound
const double MCMDiffusivityUpperBound = 3.0e-3;

//! Radial diffusivity upper bound
const double MCMRadialDiffusivityUpperBound = 1.0e-3;

//! Free water diffusivity lower bound
const double MCMFreeWaterDiffusivityLowerBound = 2.0e-3;

//! Free water diffusivity upper bound
const double MCMFreeWaterDiffusivityUpperBound = 4.0e-3;

//! Isotropic restricted water diffusivity lower bound
const double MCMIsotropicRestrictedWaterDiffusivityLowerBound = 0.25e-4;

//! Isotropic restricted water diffusivity upper bound
const double MCMIsotropicRestrictedWaterDiffusivityUpperBound = 2.0e-3;

//! Polar angle upper bound (used in tensor for now)
const double MCMPolarAngleUpperBound = M_PI;

//! Azimuth angle upper bound
const double MCMAzimuthAngleUpperBound = 2.0 * M_PI;

//! Concentration upper bound (used in NODDI and DDI)
const double MCMConcentrationUpperBound = 128.0;

//! Fraction upper bound (intra/extra axonal)
const double MCMFractionUpperBound = 1.0;

//! Tissue radius lower bound (in Stanisz for now)
const double MCMTissueRadiusLowerBound = 1.0e-4;

//! Tissue radius upper bound (in Stanisz for now)
const double MCMTissueRadiusUpperBound = 4.01e-2;

}
