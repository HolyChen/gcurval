#pragma once

#include "../config.h"

#define EIGEN_FAST_MATH 1
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <type_traits>
#include <Eigen/Eigen>

namespace gcurval
{

template <typename Dt>
using Vector3X = Eigen::Matrix<Dt, 3, 1>;

template <typename Dt>
using Vector2X = Eigen::Matrix<Dt, 2, 1>;

template <typename Dt, int M>
using VectorX = Eigen::Matrix<Dt, M, 1>;

template <typename T>
struct wider_real_type
{
};

template <>
struct wider_real_type<float>
{
    using type = double;
};

template <>
struct wider_real_type<double>
{
    // long double should be used here!
    // but cuda treats long double as double now.
    using type = double;
};

template <>
struct wider_real_type<long double>
{
    // no wider
};

template <typename T>
using wider_real_type_t = typename wider_real_type<T>::type;

template <typename T>
void free_s(T*& ptr)
{
    if (ptr)
    {
        delete ptr;
        ptr = nullptr;
    }
}

template <typename T>
void freeArray_s(T*& ptr)
{
    if (ptr)
    {
        delete[] ptr;
        ptr = nullptr;
    }
}

enum GcurvalEnum
{
    gcurvalDevice,
    gcurvalHost,
    gcurvalReverseReturnUs,
    gcurvalReverseReturnPoints
};

}

#if __cplusplus >= 201703L

#define RESULT_OF std::invoke_result
#define RESULT_OF_T std::invoke_result_t

#else

#define RESULT_OF std::result_of
#define RESULT_OF_T std::result_of_t

#endif // !C++17

#ifdef USE_CUDA

#define GCURVAL_HD_FUNC __host__ __device__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#else

#define GCURVAL_HD_FUNC 

#endif // !__CUDA_ACC__

namespace gcurval
{

template <typename T>
GCURVAL_HD_FUNC T abs(T v);

template <>
GCURVAL_HD_FUNC float abs<float>(float v)
{
    return fabsf(v);
}

template <>
GCURVAL_HD_FUNC double abs<double>(double v)
{
    return fabs(v);
}

template <>
GCURVAL_HD_FUNC long double abs<long double>(long double v)
{
#ifdef __CUDA_ARCH__
    return (long double)(fabs((double)v));
#else
    return std::fabsl(v);
#endif // !__CUDA_ARCH__
}

// precisely summation
template <typename ResultT, typename Dt>
GCURVAL_HD_FUNC ResultT precisely_summation(ResultT sum, Dt a, ResultT residual);

template <>
GCURVAL_HD_FUNC double precisely_summation<double, float>(double sum, float a, double residual)
{
    // As float value, the we use a wider type, namely double, to accmuluate
    return sum + a;
}

template <>
GCURVAL_HD_FUNC double precisely_summation<double, double>(double sum, double a, double residual)
{
    // As type value, no wider type anymore now in CUDA or MSVC, use Kahan summation
    // See: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    return sum + (a - residual);
}

template <typename ResultT, typename Dt>
GCURVAL_HD_FUNC void precisely_summation_ref(ResultT& sum, Dt a, ResultT& residual);

template <>
GCURVAL_HD_FUNC void precisely_summation_ref<double, float>(double& sum, float a, double& residual)
{
    // As float value, the we use a wider type, namely double, to accmuluate
    sum += a;
}

template <>
GCURVAL_HD_FUNC void precisely_summation_ref<double, double>(double& sum, double a, double& residual)
{
    // As type value, no wider type anymore now in CUDA or MSVC, use Kahan summation
    // See: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    double y = a - residual;
    double t = sum + y;
    residual = (t - sum) - y;
    sum = t;
}

}