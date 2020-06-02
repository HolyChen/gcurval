#pragma once

#ifdef USE_CUDA

#include "detail/cuda/arclength_calculate.h"

namespace gcurval
{

template <typename Curve,
    typename CurveTraits,
    int max_recursive_depth = gcurval::MaxRecursiveDepth<typename Curve::DataType>::value,
    int n_subdivided = gcurval::N_SUBDIVIDED,
    typename BufferManager = 
    BufferManagerBase<typename Curve::DataType, max_recursive_depth,
    typename CurveTraits::ForwardSpace, typename CurveTraits::BackwardSpace>>
using ArclengthCalculateGpu =
    ArclengthCalculateCuda<Curve, CurveTraits, max_recursive_depth, n_subdivided, BufferManager>;

}

#endif // USE_CUDA

#include "detail/cpu/arclength_calculate.h"
