#pragma once

#include <gcurval/gcurval.h>

template <typename Dt, typename Vector = gcurval::Vector2X<Dt>>
GCURVAL_HD_FUNC Vector point(Dt theta, Dt a, Dt b)
{
    return Vector(a * std::cos(theta), b * std::sin(theta));
}

template <typename Dt, typename Vector = gcurval::Vector2X<Dt>>
GCURVAL_HD_FUNC Vector derv_o1(Dt theta, Dt a, Dt b)
{
    return Vector(-1.0 * a * std::sin(theta), b * std::cos(theta));
}

template <typename Dt>
struct CurvesEllipse : gcurval::CurvesBase<Dt, gcurval::Vector2X<Dt>>
{
    using PointType = gcurval::Vector2X<Dt>;

public:
    Dt *a = nullptr;
    Dt *b = nullptr;

public:
    GCURVAL_HD_FUNC explicit constexpr CurvesEllipse(
        int n_curve = 0,
        Dt* a = nullptr,
        Dt* b = nullptr)
        : CurvesBase<Dt, PointType>(n_curve), a(a), b(b)
    {
    }
};


template <typename Dt>
struct ParaPackEllipse : gcurval::ParaPackBase<Dt>
{
    Dt a;
    Dt b;

public:
    GCURVAL_HD_FUNC ParaPackEllipse() = default; // cuda shared variable need default constructor

    GCURVAL_HD_FUNC explicit constexpr ParaPackEllipse(
        Dt begin, Dt end, int cid, int sid, Dt a = Dt{}, Dt b = Dt{})
        : gcurval::ParaPackBase<Dt>{ begin, end, cid, sid },
        a(a), b(b)
    {
    }
};


template <typename Dt>
struct ParaPackParserEllipse : gcurval::ParaPackParserBase<Dt, ParaPackEllipse<Dt>>
{
    using Base = gcurval::ParaPackParserBase<Dt, ParaPackEllipse<Dt>>;

    GCURVAL_HD_FUNC auto operator()(
        const CurvesEllipse<Dt>& curve, int cid, int segid, bool need_interval = false) const
    {
        return ParaPackEllipse<Dt> { 0, 2 * EIGEN_PI, cid, 0, curve.a[cid], curve.b[cid] };
    }
};

template <typename Dt>
struct GetPointEllipse
{
    GCURVAL_HD_FUNC typename CurvesEllipse<Dt>::PointType
        operator()(Dt u, const ParaPackEllipse<Dt>& pk) const
    {
        return point(u, pk.a, pk.b);
    }
};

template <typename Dt>
struct GetDerivativeEllipse
{
    GCURVAL_HD_FUNC typename CurvesEllipse<Dt>::PointType
        operator()(Dt u, const ParaPackEllipse<Dt>& pk) const
    {
        return derv_o1(u, pk.a, pk.b);
    }
};

template <typename Dt>
struct CurveTraitsEllipse : public gcurval::CurveTraitsBase<CurvesEllipse<Dt>>
{
    using ParaPackParser = ParaPackParserEllipse<Dt>;
    using ParaPack = ParaPackEllipse<Dt>;
    using GetPoint = GetPointEllipse<Dt>;
    using GetDerivative = GetDerivativeEllipse<Dt>;
    //using CalculateFunction = CalculateFunctionVincentForseyRuleImpl<Dt, ParaPack, GetPoint>;
    using CalculateFunction = gcurval::CalculateFunctionGaussLegendreQuadratureImpl<Dt, ParaPack, GetDerivative>;
    //using CalculateFunction = gcurval::CalculateFunctionParameterDebugMethodImpl<Dt, ParaPack, GetDerivative>;
};

#ifdef USE_CUDA

template <typename Dt,
    int max_recursive_depth = gcurval::MaxRecursiveDepth<Dt>::value,
    int n_subdivided = 64u>
    class ArclengthCalculateGpu :
    public gcurval::ArclengthCalculateGpu<
    CurvesEllipse<Dt>,
    CurveTraitsEllipse<Dt>,
    max_recursive_depth,
    n_subdivided>
{
private:
    using Base = gcurval::ArclengthCalculateGpu<
        CurvesEllipse<Dt>,
        CurveTraitsEllipse<Dt>,
        max_recursive_depth,
        n_subdivided>;

    using Vector = gcurval::Vector2X<Dt>;

public:
    explicit ArclengthCalculateGpu(int n_curve, Dt *a, Dt *b)
        : Base(n_curve, n_curve)
    {
        gMalloc(&d_curve.a, sizeof(Dt) * n_curve);
        gMalloc(&d_curve.b, sizeof(Dt) * n_curve);

        cudaMemcpy(d_curve.a, a, sizeof(Dt) * n_curve, cudaMemcpyHostToDevice);
        cudaMemcpy(d_curve.b, b, sizeof(Dt) * n_curve, cudaMemcpyHostToDevice);
    }

    ~ArclengthCalculateGpu()
    {
        gFree_s(d_curve.a);
        gFree_s(d_curve.b);
    }
};

#endif // USE_CUDA


template <typename Dt,
    int max_recursive_depth = gcurval::MaxRecursiveDepth<Dt>::value>
    class ArclengthCalculateCpu :
    public gcurval::ArclengthCalculateCpu<
    CurvesEllipse<Dt>,
    CurveTraitsEllipse<Dt>,
    max_recursive_depth>
{
private:
    using Base = gcurval::ArclengthCalculateCpu<
        CurvesEllipse<Dt>,
        CurveTraitsEllipse<Dt>,
        max_recursive_depth>;

    using Vector = gcurval::Vector2X<Dt>;

public:
    explicit ArclengthCalculateCpu(int n_curve, Dt *a, Dt *b)
        : Base(n_curve, n_curve)
    {
        curve.a = new Dt[n_curve];
        curve.b = new Dt[n_curve];

        std::copy(a, a + n_curve, curve.a);
        std::copy(b, b + n_curve, curve.b);
    }

    ~ArclengthCalculateCpu()
    {
        gcurval::freeArray_s(curve.a);
        gcurval::freeArray_s(curve.b);
    }
};
