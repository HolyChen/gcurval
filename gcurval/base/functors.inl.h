#pragma once

#include "util.h"
#include "structs.inl.h"

namespace gcurval
{

template <typename Dt, typename PointType, typename Rt = ParaPackBase<Dt>>
struct ParaPackParserBase
{
    using result_type = Rt;

    GCURVAL_HD_FUNC auto operator()(
        const CurvesBase<Dt, PointType>& curve, int cid, int sid, bool need_interval = false) const
    {
        return Rt{ 0.0, 0.0, cid, sid, 0, 0 };
    }
};

template <typename Dt>
struct ParaPackSaverBase
{
    template <typename ParaPack>
    GCURVAL_HD_FUNC void operator()(int pos,
        ForwardSpaceBase<Dt>& fs,
        BackwardSpaceBase<Dt>& bs,
        Dt begin, Dt end, int tid, Dt al, const ParaPack& old_pack) const
    {
        fs.begin[pos] = begin;
        fs.end[pos] = end;
        fs.cid[pos] = old_pack.cid;
        fs.sid[pos] = old_pack.sid;
        bs.ptid[pos] = tid;
        bs.pal[pos] = al;
    }
};

template <typename Dt>
struct GaussLegendreQuadrature
{
    static constexpr int N_GAUSS = 3;
    using GLC = GaussLegendreCoeff<Dt>;

    template <typename GetDerivative, typename ParaPack>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end,
        const GetDerivative& func_v, const ParaPack& pack)
    {
        using Rt = decltype(func_v(begin, pack));

        Dt k = (end - begin) / 2.0;
        Dt b = (end + begin) / 2.0;

        Dt accumulate = 0.0;

#pragma unroll(N_GAUSS)
        for (int j = 0; j < N_GAUSS; j++)
        {
            Dt u = k * GLC::x(N_GAUSS, j) + b;
            accumulate += GLC::w(N_GAUSS, j) *
                _getValue<Dt, Rt>{}(u, func_v, pack);
        }
        accumulate *= k;

        return accumulate;
    }

    template <typename GetDerivative, typename ParaPack, typename SharedMem>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end,
        const GetDerivative& func_v, const ParaPack& pack, SharedMem& buffer)
    {
        using Rt = decltype(func_v(begin, pack, buffer));

        Dt k = (end - begin) / 2.0;
        Dt b = (end + begin) / 2.0;

        Dt accumulate = 0.0;

#pragma unroll(N_GAUSS)
        for (int j = 0; j < N_GAUSS; j++)
        {
            Dt u = k * GLC::x(N_GAUSS, j) + b;
            accumulate += GLC::w(N_GAUSS, j) * 
                _getValueSharedMem<Dt, Rt>{}(u, func_v, pack, buffer);
        }
        accumulate *= k;

        return accumulate;
    }

private:
    template <typename T, typename Rt, bool = std::is_same<T, Rt>::value>
    struct _getValue;

    template <typename T, typename Rt>
    struct _getValue<T, Rt, true>
    {
        template <typename GetDerivative, typename ParaPack>
        GCURVAL_HD_FUNC Dt operator()(Dt u,
            const GetDerivative& func_v, const ParaPack& pack)
        {
            return func_v(u, pack);
        }
    };

    template <typename T, typename Rt>
    struct _getValue<T, Rt, false>
    {
        template <typename GetDerivative, typename ParaPack>
        GCURVAL_HD_FUNC decltype(std::declval<Rt>().norm()) operator()(Dt u,
            const GetDerivative& func_v, const ParaPack& pack)
        {
            return func_v(u, pack).norm();
        }
    };

    template <typename T, typename Rt, bool = std::is_same<T, Rt>::value>
    struct _getValueSharedMem;

    template <typename T, typename Rt>
    struct _getValueSharedMem<T, Rt, true>
    {
        template <typename GetDerivative, typename ParaPack, typename SharedMem>
        GCURVAL_HD_FUNC Dt operator()(Dt u,
            const GetDerivative& func_v, const ParaPack& pack, SharedMem& buffer)
        {
            return func_v(u, pack, buffer);
        }
    };

    template <typename T, typename Rt>
    struct _getValueSharedMem<T, Rt, false>
    {
        template <typename GetDerivative, typename ParaPack, typename SharedMem>
        GCURVAL_HD_FUNC decltype(std::declval<Rt>().norm()) operator()(Dt u,
            const GetDerivative& func_v, const ParaPack& pack, SharedMem& buffer)
        {
            return func_v(u, pack, buffer).norm();
        }
    };
};

// -- point based --

template <typename Dt>
struct PolylineMethod
{
    template <typename GetPoint, typename ParaPack>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end, const GetPoint& curve_point, const ParaPack& pack)
    {
        auto v0 = curve_point(begin, pack);
        auto v2 = curve_point(end, pack);

        return (v2 - v0).norm();
    }

    template <typename GetPoint, typename ParaPack, typename SharedMem>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end,
        const GetPoint& curve_point, const ParaPack& pack, SharedMem& buffer)
    {
        auto v0 = curve_point(begin, pack, buffer);
        auto v2 = curve_point(end, pack, buffer);

        return (v2 - v0).norm();
    }
};

template <typename Dt>
struct SimpsonMethod
{
    template <typename GetPoint, typename ParaPack>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end, const GetPoint& curve_point, const ParaPack& pack)
    {
        Dt mid = (begin + end) / Dt(2.0);

        auto v0 = curve_point(begin, pack);
        auto v1 = curve_point(mid, pack);
        auto v2 = curve_point(end, pack);

        Dt al = 1.0 / 6.0 * (
            (-3.0 * v0 + 4.0 * v1 - v2).norm() +
            4.0 * (v2 - v0).norm() +
            (v0 - 4.0 * v1 + 3.0 * v2).norm());

        return al;
    }

    template <typename GetPoint, typename ParaPack, typename SharedMem>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end,
        const GetPoint& curve_point, const ParaPack& pack, SharedMem& buffer)
    {
        Dt mid = (begin + end) / Dt(2.0);

        auto v0 = curve_point(begin, pack, buffer);
        auto v1 = curve_point(mid, pack, buffer);
        auto v2 = curve_point(end, pack, buffer);

        Dt al = 1.0 / 6.0 * (
            (-3.0 * v0 + 4.0 * v1 - v2).norm() +
            4.0 * (v2 - v0).norm() +
            (v0 - 4.0 * v1 + 3.0 * v2).norm());

        return al;
    }
};

template <typename Dt>
struct VincentForseyRule
{
    template <typename GetPoint, typename ParaPack>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end, const GetPoint& curve_point, const ParaPack& pack)
    {
        Dt mid = (begin + end) / Dt(2.0);

        auto v0 = curve_point(begin, pack);
        auto v1 = curve_point(mid, pack);
        auto v2 = curve_point(end, pack);

        Dt al = 4.0 / 3.0 * ((v0 - v1).norm() + (v2 - v1).norm()) - 1.0 / 3.0 * (v2 - v0).norm();

        return al;
    }

    template <typename GetPoint, typename ParaPack, typename SharedMem>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end,
        const GetPoint& curve_point, const ParaPack& pack, SharedMem& buffer)
    {
        Dt mid = (begin + end) / Dt(2.0);

        auto v0 = curve_point(begin, pack, buffer);
        auto v1 = curve_point(mid, pack, buffer);
        auto v2 = curve_point(end, pack, buffer);

        Dt al = 4.0 / 3.0 * ((v0 - v1).norm() + (v2 - v1).norm()) - 1.0 / 3.0 * (v2 - v0).norm();

        return al;
    }
};

// ------------- for debug -------------

template <typename Dt>
struct ParameterDebugMethod
{
    template <typename GetPoint, typename ParaPack>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end, const GetPoint& curve_point, const ParaPack& pack)
    {
        return abs(end - begin);
    }

    template <typename GetPoint, typename ParaPack, typename SharedMem>
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end,
        const GetPoint& curve_point, const ParaPack& pack, SharedMem& buffer)
    {
        return abs(end - begin);
    }
};


// ----------- method end ------------

template <typename Curve>
struct CurveLocatorBase
{
    GCURVAL_HD_FUNC constexpr CurveIntervalPos operator()(const Curve&, int cid)
    {
        return CurveIntervalPos{ cid, cid + 1 };
    }
};

template <typename SharedMemAllocator>
struct DoArclengthCalculateStep
{
    template <typename Dt, typename ParaPack, typename CalculateFunction>
    GCURVAL_HD_FUNC Dt operator()(const CalculateFunction& calculateFunction,
        Dt begin, Dt end, const ParaPack& pack, uint8_t *buffer)
    {
        return calculateFunction(begin, end, pack, SharedMemAllocator{}(buffer));
    }
};

template <>
struct DoArclengthCalculateStep<NoSharedMemAllocator>
{
    template <typename Dt, typename ParaPack, typename CalculateFunction>
    GCURVAL_HD_FUNC Dt operator()(const CalculateFunction& calculateFunction,
        Dt begin, Dt end, const ParaPack& pack, uint8_t *buffer)
    {
        return calculateFunction(begin, end, pack);
    }
};

template <typename SharedMemAllocator>
struct DoGetPointStep
{
    template <typename Dt, typename ParaPack, typename GetPoint>
    GCURVAL_HD_FUNC
    RESULT_OF_T<GetPoint(Dt, ParaPack, RESULT_OF_T<SharedMemAllocator(uint8_t*)>)>
    operator()(const GetPoint& getPoint,
        Dt u, const ParaPack& pack, uint8_t *buffer)
    {
        return getPoint(u, pack, SharedMemAllocator{}(buffer));
    }
};

template <>
struct DoGetPointStep<NoSharedMemAllocator>
{
    template <typename Dt, typename ParaPack, typename GetPoint>
    GCURVAL_HD_FUNC RESULT_OF_T<GetPoint(Dt, ParaPack)>
    operator()(const GetPoint& getPoint,
        Dt u, const ParaPack& pack, uint8_t *buffer)
    {
        return getPoint(u, pack);
    }
};

#define MAKE_COMPUTE_FUNCTION_IMPL(METHOD_NAME) \
template <typename Dt, typename ParaPack, typename GetP> \
struct CalculateFunction ## METHOD_NAME ## Impl \
{ \
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end, const ParaPack& pack) const \
    { \
        return METHOD_NAME<Dt>{}(begin, end, GetP{}, pack); \
    } \
\
    template <typename SharedMem> \
    GCURVAL_HD_FUNC Dt operator()(Dt begin, Dt end, \
        const ParaPack& pack, SharedMem& buffer) const \
    { \
        return METHOD_NAME<Dt>{}(begin, end, GetP{}, pack, buffer); \
    } \
}; \
\

MAKE_COMPUTE_FUNCTION_IMPL(GaussLegendreQuadrature)
MAKE_COMPUTE_FUNCTION_IMPL(PolylineMethod)
MAKE_COMPUTE_FUNCTION_IMPL(SimpsonMethod)
MAKE_COMPUTE_FUNCTION_IMPL(VincentForseyRule)
MAKE_COMPUTE_FUNCTION_IMPL(ParameterDebugMethod)

#undef MAKE_COMPUTE_FUNCTION_IMPL

}