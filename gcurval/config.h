#pragma once

// There are some user defined macros:
//
// GCURVAL_N_SUBDIVIDED:
//      to define how many subsegment a segment to subdivide.
//      The default value is 64, 2 size of a warp.
// GCURVAL_MAX_ERROR:
//      to define the max error when arc length calculation and arc length sampling
//      The default values are 1e-6f for float, and 1e-14 for double.
//      Change the default value is NOT recommended.
// GCURVAL_MAX_RECURSIVE_DEPTH:
//      to define the max recursive or iteration depth for the algorithms.
//      The default values are 14 for float, 23 for double, the maximum is 32
//      Change the default value is NOT recommended.

namespace gcurval
{
    // how many subsegment a segment to subdivide
#ifdef GCURVAL_N_SUBDIVIDED
    constexpr int N_SUBDIVIDED = GCURVAL_N_SUBDIVIDED;
#else
    constexpr int N_SUBDIVIDED = 64;
#endif // GCURVAL_N_SUBDIVIDED

    // the \xi and \epsilon for float and double
#ifdef GCURVAL_MAX_ERROR
    constexpr float MAX_ERROR_FLOAT = GCURVAL_MAX_ERROR;
    constexpr float MAX_ERROR_DOUBLE = GCURVAL_MAX_ERROR;
#else
    constexpr float MAX_ERROR_FLOAT = 1e-6f;
    constexpr double MAX_ERROR_DOUBLE = 1e-14;
#endif // GCURVAL_MAX_ERROR_FLOAT

    // the max recrusive/iteration depth for float and double
    template <typename Dt> struct MaxRecursiveDepth;
#ifdef GCURVAL_MAX_RECURSIVE_DEPTH
    template <> struct MaxRecursiveDepth<float> { static constexpr int value = GCURVAL_MAX_RECURSIVE_DEPTH; };
    template <> struct MaxRecursiveDepth<double> { static constexpr int value = GCURVAL_MAX_RECURSIVE_DEPTH; };
#else
    template <> struct MaxRecursiveDepth<float> { static constexpr int value = 14; };
    template <> struct MaxRecursiveDepth<double> { static constexpr int value = 23; };
#endif // GCURVAL_MAX_RECURSIVE_DEPTH

}