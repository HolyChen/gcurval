#pragma once

//#include <cnmem.h>

#include <gcurval/gcurval.h>
#include "structs.h"

template <typename Dt, typename PointType, int degree = 3>
struct CurveTraitsBSpline : gcurval::CurveTraitsBase<CurvesBSpline<Dt, PointType, degree>>
{
    using ParaPack = ParaPackBSpline<Dt, PointType>;
    using ParaPackParser = ParaPackParserBSpline<Dt, PointType, degree>;
    using SharedMemAllocator = SharedMemAllocatorBSpline<Dt, degree>;
    using CurveLocator = CurveLocatorBSpline<Dt, PointType, degree>;

    using GetPoint = GetPointBSpline<Dt, PointType, degree>;
    using GetDerivative = GetDerivativeBSpline<Dt, PointType, degree>;
    //using CalculateFunction = CalculateArcLengthBSpline<Dt, degree>;
    //using CalculateFunction = CalculateFunctionSimpsonMethodImpl<Dt, ParaPack, GetPoint>;
    //using CalculateFunction = gcurval::CalculateFunctionVincentForseyRuleImpl<Dt, ParaPack, GetPoint>;
    using CalculateFunction = gcurval::CalculateFunctionGaussLegendreQuadratureImpl<Dt, ParaPack, GetDerivative>;
    //using CalculateFunction = gcurval::CalculateFunctionParameterDebugMethodImpl<Dt, ParaPack, GetPoint>;
};

#ifdef USE_CUDA

template <
    typename Dt,
    typename PointType,
    int degree = 3,
    int max_recursive_depth = gcurval::MaxRecursiveDepth<Dt>::value,
    int n_subdivided = 64u>
    class ArclengthCalculateGpu : public gcurval::ArclengthCalculateGpu<
    CurvesBSpline<Dt, PointType, degree>,
    CurveTraitsBSpline<Dt, PointType, degree>,
    max_recursive_depth,
    n_subdivided>
{
private:
    using Base = gcurval::ArclengthCalculateGpu<
        CurvesBSpline<Dt, PointType, degree>,
        CurveTraitsBSpline<Dt, PointType, degree>,
        max_recursive_depth,
        n_subdivided>;

    using Vector = PointType;

public:
    explicit ArclengthCalculateGpu(int n_curve, int* knots_offset, Dt* knot_vector, Vector* ctrl_ptrs)
        : Base(n_curve, knots_offset[n_curve] - (degree * 2 + 1) * n_curve)
    {
        int n_knots = knots_offset[n_curve];
        int n_ctrl_ptrs = (knots_offset[n_curve] - (degree + 1) * n_curve);

        gMalloc(&d_curve.knot_vector, sizeof(Dt) * n_knots);
        gMalloc(&d_curve.knots_offset, sizeof(int) * (n_curve + 1));
        gMalloc(&d_curve.ctrl_ptrs, sizeof(Vector) * n_ctrl_ptrs);

        cudaMemcpy(d_curve.knot_vector, knot_vector, sizeof(Dt) * n_knots, cudaMemcpyHostToDevice);
        cudaMemcpy(d_curve.knots_offset, knots_offset, sizeof(int) * (n_curve + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_curve.ctrl_ptrs, ctrl_ptrs, sizeof(Vector) * n_ctrl_ptrs, cudaMemcpyHostToDevice);
    }

    ~ArclengthCalculateGpu()
    {
        gFree_s(d_curve.knot_vector);
        gFree_s(d_curve.knots_offset);
        gFree_s(d_curve.ctrl_ptrs);
    }
};

#endif // USE_CUDA

template <
    typename Dt,
    typename PointType,
    int degree = 3,
    int max_recursive_depth = gcurval::MaxRecursiveDepth<Dt>::value>
    class ArclengthCalculateCpu : public gcurval::ArclengthCalculateCpu<
    CurvesBSpline<Dt, PointType, degree>,
    CurveTraitsBSpline<Dt, PointType, degree>,
    max_recursive_depth>
{
private:
    using Base = gcurval::ArclengthCalculateCpu<
        CurvesBSpline<Dt, PointType, degree>,
        CurveTraitsBSpline<Dt, PointType, degree>,
        max_recursive_depth>;

    using Vector = PointType;

public:
    explicit ArclengthCalculateCpu(int n_curve, int* knots_offset, Dt* knot_vector, Vector* ctrl_ptrs)
        : Base(n_curve, knots_offset[n_curve] - (degree * 2 + 1) * n_curve)
    {
        int n_knots = knots_offset[n_curve];
        int n_ctrl_ptrs = (knots_offset[n_curve] - (degree + 1) * n_curve);

        curve.knot_vector = new Dt[n_knots];
        curve.knots_offset = new int[n_curve + 1];
        curve.ctrl_ptrs = new Vector[n_ctrl_ptrs];

        std::copy(knot_vector, knot_vector + n_knots, curve.knot_vector);
        std::copy(knots_offset, knots_offset + (n_curve + 1), curve.knots_offset);
        std::copy(ctrl_ptrs, ctrl_ptrs + n_ctrl_ptrs, curve.ctrl_ptrs);
    }

    ~ArclengthCalculateCpu()
    {
        gcurval::freeArray_s(curve.knot_vector);
        gcurval::freeArray_s(curve.knots_offset);
        gcurval::freeArray_s(curve.ctrl_ptrs);
    }
};

