#pragma once

#include "util.h"

namespace gcurval
{

#define MAKE_EPSILON(type, value, suffix, n_subdivided) \
\
__constant__ static const type d_eps_ ## suffix[32] = { \
    1 * (value / type(n_subdivided)),                         0.5 * (value / type(n_subdivided)),                       0.25 * (value / type(n_subdivided)),                           0.125 * (value / type(n_subdivided)),                    \
    0.0625 * (value / type(n_subdivided)),                    0.03125 * (value / type(n_subdivided)),                   0.015625 * (value / type(n_subdivided)),                       0.0078125 * (value / type(n_subdivided)),                \
    0.00390625 * (value / type(n_subdivided)),                0.001953125 * (value / type(n_subdivided)),               0.0009765625 * (value / type(n_subdivided)),                   0.00048828125 * (value / type(n_subdivided)),            \
    0.000244140625 * (value / type(n_subdivided)),            0.0001220703125 * (value / type(n_subdivided)),           6.103515625e-05 * (value / type(n_subdivided)),                3.0517578125e-05 * (value / type(n_subdivided)),         \
    1.52587890625e-05 * (value / type(n_subdivided)),         7.62939453125e-06 * (value / type(n_subdivided)),         3.814697265625e-06 * (value / type(n_subdivided)),             1.9073486328125e-06 * (value / type(n_subdivided)),      \
    9.5367431640625e-07 * (value / type(n_subdivided)),       4.76837158203125e-07 * (value / type(n_subdivided)),      2.384185791015625e-07 * (value / type(n_subdivided)),          1.1920928955078125e-07 * (value / type(n_subdivided)),   \
    5.9604644775390625e-08 * (value / type(n_subdivided)),    2.98023223876953125e-08 * (value / type(n_subdivided)),   1.490116119384765625e-08 * (value / type(n_subdivided)),       7.450580596923828125e-09 * (value / type(n_subdivided)), \
    3.7252902984619140625e-09 * (value / type(n_subdivided)), 1.8626451492309570312e-09 * (value / type(n_subdivided)), 9.3132257461547851562e-10 * (value / type(n_subdivided)),      4.6566128730773925781e-10 * (value / type(n_subdivided)) \
}; \
\
static const type h_eps_ ## suffix[32] = { \
    1 * (value),                         0.5 * (value),                       0.25 * (value),                           0.125 * (value),                    \
    0.0625 * (value),                    0.03125 * (value),                   0.015625 * (value),                       0.0078125 * (value),                \
    0.00390625 * (value),                0.001953125 * (value),               0.0009765625 * (value),                   0.00048828125 * (value),            \
    0.000244140625 * (value),            0.0001220703125 * (value),           6.103515625e-05 * (value),                3.0517578125e-05 * (value),         \
    1.52587890625e-05 * (value),         7.62939453125e-06 * (value),         3.814697265625e-06 * (value),             1.9073486328125e-06 * (value),      \
    9.5367431640625e-07 * (value),       4.76837158203125e-07 * (value),      2.384185791015625e-07 * (value),          1.1920928955078125e-07 * (value),   \
    5.9604644775390625e-08 * (value),    2.98023223876953125e-08 * (value),   1.490116119384765625e-08 * (value),       7.450580596923828125e-09 * (value), \
    3.7252902984619140625e-09 * (value), 1.8626451492309570312e-09 * (value), 9.3132257461547851562e-10 * (value),      4.6566128730773925781e-10 * (value) \
};


MAKE_EPSILON(float, gcurval::MAX_ERROR_FLOAT, f, gcurval::N_SUBDIVIDED)
MAKE_EPSILON(double, gcurval::MAX_ERROR_DOUBLE, d, gcurval::N_SUBDIVIDED)

#undef MAKE_EPSILON

template <typename Dt>
struct Epsilon;

template <>
struct Epsilon<float>
{
    static constexpr GCURVAL_HD_FUNC float v(int i)
    {
#ifdef __CUDA_ARCH__
        return d_eps_f[i];
#else
        return h_eps_f[i];
#endif // __CUDA_ACC__

    }

    static constexpr float REVERSE = MAX_ERROR_FLOAT;
};

template <>
struct Epsilon<double>
{
    static constexpr GCURVAL_HD_FUNC double v(int i)
    {
#ifdef __CUDA_ARCH__
        return d_eps_d[i];
#else
        return h_eps_d[i];
#endif // __CUDA_ARCH__
    }

    static constexpr double REVERSE = MAX_ERROR_DOUBLE;
};

// ------------ Gaussian Legendre Quadrature ------------

#define MAKE_GAUSSIAN_X(type, prefix, suffix) \
static const type prefix ## _gaussian_x_ ## suffix[5][5] = \
{ \
    {0., 0., 0., 0., 0.}, \
    {-0.5773502691896258, 0.5773502691896258, 0., 0., 0.}, \
    {0., -0.7745966692414834, 0.7745966692414834, 0., 0.}, \
    {-0.33998104358485626, 0.33998104358485626, -0.8611363115940526, 0.8611363115940526, 0.}, \
    {0., -0.5384693101056831, 0.5384693101056831, -0.906179845938664, 0.906179845938664} \
};

#define MAKE_GAUSSIAN_W(type, prefix, suffix) \
static const type prefix ## _gaussian_w_ ## suffix[5][5] = \
{ \
    {2., 0., 0., 0., 0.}, \
    { 1., 1., 0., 0., 0. }, \
    { 0.8888888888888889, 0.5555555555555556, 0.5555555555555556, 0., 0. }, \
    { 0.6521451548625462, 0.6521451548625462, 0.34785484513745385, 0.34785484513745385, 0. }, \
    { 0.5688888888888889, 0.47862867049936647, 0.47862867049936647, 0.23692688505618908, 0.23692688505618908 }, \
};

#define MAKE_GAUSSIAN_ALL(type, suffix) \
MAKE_GAUSSIAN_X(type, h, suffix) \
MAKE_GAUSSIAN_W(type, h, suffix) \
__constant__ MAKE_GAUSSIAN_X(type, d, suffix) \
__constant__ MAKE_GAUSSIAN_W(type, d, suffix)

MAKE_GAUSSIAN_ALL(float, f)
MAKE_GAUSSIAN_ALL(double, d)

#undef MAKE_GAUSSIAN_X
#undef MAKE_GAUSSIAN_W

template <typename Dt>
struct GaussLegendreCoeff;

template <>
struct GaussLegendreCoeff<float>
{
    static constexpr GCURVAL_HD_FUNC float x(int order, int i)
    {
#ifdef __CUDA_ARCH__
        return d_gaussian_x_f[order - 1][i];
#else
        return h_gaussian_x_f[order - 1][i];
#endif // __CUDA_ARCH__

    }

    static constexpr GCURVAL_HD_FUNC float w(int order, int i)
    {
#ifdef __CUDA_ARCH__
        return d_gaussian_w_f[order - 1][i];
#else
        return h_gaussian_w_f[order - 1][i];
#endif // __CUDA_ARCH__   
    }
};

template <>
struct GaussLegendreCoeff<double>
{
    static constexpr GCURVAL_HD_FUNC double x(int order, int i)
    {
#ifdef __CUDA_ARCH__
        return d_gaussian_x_d[order - 1][i];
#else
        return h_gaussian_x_d[order - 1][i];
#endif // __CUDA_ARCH__

    }

    static constexpr GCURVAL_HD_FUNC double w(int order, int i)
    {
#ifdef __CUDA_ARCH__
        return d_gaussian_w_d[order - 1][i];
#else
        return h_gaussian_w_d[order - 1][i];
#endif // __CUDA_ARCH__   
    }
};

}