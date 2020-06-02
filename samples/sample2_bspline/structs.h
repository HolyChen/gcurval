#pragma once

#include <gcurval/base.h>
#include "BSpline.h"

template <typename Dt, typename Pt, int degree>
struct CurvesBSpline : gcurval::CurvesBase<Dt, Pt>
{
    using PointType = Pt;

public:
    int *knots_offset = nullptr;
    Dt *knot_vector = nullptr;
    PointType *ctrl_ptrs = nullptr;

public:
    GCURVAL_HD_FUNC constexpr CurvesBSpline(
        int n_curve = 0,
        int* knots_offset = nullptr,
        Dt* knot_vector = nullptr,
        PointType* ctrl_ptrs = nullptr)
        : CurvesBase<Dt, PointType>(n_curve), knots_offset(knots_offset), knot_vector(knot_vector), ctrl_ptrs(ctrl_ptrs)
    {
    }
};

template <typename Dt>
struct SharedMemBSpline
{
    Dt* func_value_buf;
    Dt* left_buf;
    Dt* right_buf;
};

template <typename Dt, int degree>
struct SharedMemAllocatorBSpline
{
    GCURVAL_HD_FUNC SharedMemBSpline<Dt> operator()(uint8_t* buffer) const
    {
#ifdef __CUDA_ARCH__
        return
        {
            &((Dt*)buffer)[(degree + 1) * 3 * threadIdx.x + 0],
            &((Dt*)buffer)[(degree + 1) * 3 * threadIdx.x + (degree + 1)],
            &((Dt*)buffer)[(degree + 1) * 3 * threadIdx.x + (degree + 1) * 2]
        };
#else
        return
        {
            &((Dt*)buffer)[0],
            &((Dt*)buffer)[degree + 1],
            &((Dt*)buffer)[(degree + 1) * 2]
        };
#endif // __CUDA_ARCH__
    }

#ifdef __CUDA_ARCH__
    static int size(dim3 blockDim)
    {
        return blockDim.x * (degree + 1) * 3 * sizeof(Dt);
    }
#endif // __CUDA_ARCH__

    static int size(int blockDim = 1)
    {
        return blockDim * (degree + 1) * 3 * sizeof(Dt);
    }
};

template <typename Dt>
struct SharedMemAllocatorBSpline<Dt, 3>
{
    GCURVAL_HD_FUNC SharedMemBSpline<Dt> operator()(uint8_t* buffer) const
    {
#ifdef __CUDA_ARCH__
        return
        {
            &((Dt*)buffer)[13 * threadIdx.x + 0],
            &((Dt*)buffer)[13 * threadIdx.x + 4],
            &((Dt*)buffer)[13 * threadIdx.x + 8]
        };
#else
        return
        {
            &((Dt*)buffer)[0],
            &((Dt*)buffer)[4],
            &((Dt*)buffer)[8]
        };
#endif // __CUDA_ARCH__
    }

#ifdef __CUDA_ARCH__
    static int size(dim3 blockDim)
    {
        return blockDim.x * 13 * sizeof(Dt);
    }
#endif // __CUDA_ARCH__

    static int size(int blockDim = 1)
    {
        return blockDim * 13 * sizeof(Dt);
    }
};

template <typename Dt, typename Pt>
struct ParaPackBSpline : gcurval::ParaPackBase<Dt>
{
    using PointType = Pt;

    int span;
    Dt* knot_vector;
    PointType* ctrl_ptrs;

public:
    GCURVAL_HD_FUNC ParaPackBSpline() = default;

    GCURVAL_HD_FUNC  constexpr ParaPackBSpline(Dt begin, Dt end, int cid, int segid,
        int span = 0, Dt* knot_vector = nullptr, PointType* ctrl_ptrs = nullptr)
        : ParaPackBase<Dt>{ begin, end, cid, segid },
        span(span), knot_vector(knot_vector), ctrl_ptrs(ctrl_ptrs)
    {
    }
};


template <typename Dt, typename PointType, int degree>
struct ParaPackParserBSpline : gcurval::ParaPackParserBase<Dt, ParaPackBSpline<Dt, PointType>>
{
    GCURVAL_HD_FUNC auto operator()(
        const CurvesBSpline<Dt, PointType, degree>& curve, int cid, int segid, bool need_interval = false) const
    {
        int knot_offset = curve.knots_offset[cid];
        int knot_offset_seg = knot_offset + segid + degree;

        if (need_interval)
        {
            return ParaPackBSpline<Dt, PointType> {
                curve.knot_vector[knot_offset_seg], curve.knot_vector[knot_offset_seg + 1],
                    cid, segid, segid + degree, curve.knot_vector + knot_offset,
                    curve.ctrl_ptrs + knot_offset - (degree + 1) * cid };
        }
        else
        {
            return ParaPackBSpline<Dt, PointType> {
                0.0, 0.0,
                    cid, segid, segid + degree, curve.knot_vector + knot_offset,
                    curve.ctrl_ptrs + knot_offset - (degree + 1) * cid };
        }
    }
};

// Locate the start and end + 1 position in the input dataset, that is curves.
template <typename Dt, typename PointType, int degree>
struct CurveLocatorBSpline
{
    GCURVAL_HD_FUNC constexpr gcurval::CurveIntervalPos operator()(
        const CurvesBSpline<Dt, PointType, degree>& curves, int cid) const
    {
        int begin = curves.knots_offset[cid] - (2 * degree + 1) * cid;
        int end = curves.knots_offset[cid + 1] - (2 * degree + 1) * (cid + 1);
        return gcurval::CurveIntervalPos{ begin ,end };
    }
};

template <typename Dt, typename PointType, int degree>
struct GetPointBSpline
{
    GCURVAL_HD_FUNC PointType operator()(Dt u, const ParaPackBSpline<Dt, PointType>& pk, SharedMemBSpline<Dt>& buffer) const
    {
        b_spline_func_value(u, pk.span, pk.knot_vector,
            buffer.func_value_buf, buffer.left_buf, buffer.right_buf);
        return to_point<Dt, PointType, degree>(pk.span, pk.ctrl_ptrs, buffer.func_value_buf);
    }
};

template <typename Dt, typename PointType, int degree>
struct GetDerivativeBSpline
{
    GCURVAL_HD_FUNC PointType operator()(Dt u, const ParaPackBSpline<Dt, PointType>& pk, SharedMemBSpline<Dt>& buffer) const
    {
        b_spline_derv_o1(u, pk.span, pk.knot_vector,
            buffer.func_value_buf, buffer.left_buf, buffer.right_buf);
        return to_point<Dt, PointType, degree>(pk.span, pk.ctrl_ptrs, buffer.func_value_buf);
    }
};
