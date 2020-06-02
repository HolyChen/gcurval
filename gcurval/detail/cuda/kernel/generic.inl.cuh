#pragma once

#include "../util.h"
#include "../../../base.h"

namespace gcurval
{


template <typename Curve, typename CurveTraits, int n_subdivide = 64u>
__global__ void kn_curveLengthTableInit(
    Curve curve, typename CurveTraits::ForwardSpace fs, typename CurveTraits::BackwardSpace bs)
{
    using DataType = typename CurveTraits::DataType;
    using CurveLocator = typename CurveTraits::CurveLocator;
    using ParaPack = typename CurveTraits::ParaPack;
    using ParaPackParser = typename CurveTraits::ParaPackParser;
    using ParaPackSaver = typename CurveTraits::ParaPackSaver;

    __shared__ ParaPack pack_seg;
    __shared__ DataType delta;

    auto cid = blockIdx.x;
    auto curve_pos = CurveLocator()(curve, cid);
    auto n_seg = curve_pos.end - curve_pos.begin;
    auto store_begin = curve_pos.begin * n_subdivide;

    // assume the blockDim.x == n_subdivide
    for (auto segid = 0; segid < n_seg; segid++)
    {
        if (threadIdx.x == 0)
        {
            pack_seg = ParaPackParser()(curve, cid, segid, true); // need retern interval
            delta = (pack_seg.end - pack_seg.begin) / DataType(n_subdivide);
        }
        __syncthreads();

        int store_pos = store_begin + segid * n_subdivide + threadIdx.x;

        ParaPackSaver()(store_pos, fs, bs,
            pack_seg.begin + delta * threadIdx.x,
            pack_seg.begin + delta * (threadIdx.x + 1),
            0,
            0.0,
            pack_seg);
        __syncthreads();
    }
}


// ----------- merge -------------
template <int length, typename Dt>
__device__ void inclusiveScan(int lane, Dt *buf)
{
    // Use Brent-Kung adder algorithm
#pragma unroll
    for (int stride = 1; stride < length; stride *= 2)
    {
        __syncthreads();

        int index = (lane + 1) * 2 * stride - 1;

        if (index < length)
        {
            buf[index] += buf[index - stride];
        }
    }

#pragma unroll
    for (int stride = length / 4; stride > 0; stride /= 2)
    {
        __syncthreads();

        int index = (lane + 1) * stride * 2 - 1;
        if (index + stride < length)
        {
            buf[index + stride] += buf[index];
        }
    }
}


template <typename Dt, int n_subdivide = 64u>
__global__ void kn_mergeSubdividedIntoSeg(Dt *arclength_subdivided, Dt *arclength_seg, int n_task)
{
    extern __shared__ Dt local_cache[];

    auto stripe = blockDim.x * gridDim.x;

    for (int load_start = blockIdx.x * blockDim.x; load_start < n_task; load_start += stripe)
    {
        if (load_start + threadIdx.x < n_task)
        {
            local_cache[threadIdx.x] = arclength_subdivided[load_start + threadIdx.x];
        }

        inclusiveScan<n_subdivide>(threadIdx.x % n_subdivide, &local_cache[threadIdx.x / n_subdivide * n_subdivide]);

        __syncthreads();

        // store prefix sum into arclength_subdivided
        if (load_start + threadIdx.x < n_task)
        {
            arclength_subdivided[load_start + threadIdx.x] = local_cache[threadIdx.x];
        }

        int store_start = load_start / n_subdivide;

        // extract segs' arclength and store them into arclength_seg
        if (threadIdx.x < blockDim.x / n_subdivide && (store_start + threadIdx.x) * n_subdivide < n_task)
        {
            arclength_seg[store_start + threadIdx.x] = local_cache[(threadIdx.x + 1) * n_subdivide - 1];
        }
    }
}

template <typename Dt, typename Curve, typename CurveLocator>
__global__ void kn_mergeSeg(Curve curve, Dt *arclength_seg, CurveLocator locator)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < curve.n_curve)
    {
        auto pos = locator(curve, tid);
        int begin = pos.begin;
        int end = pos.end;

        wider_real_type_t<Dt> sum = arclength_seg[begin], residual = Dt{};
        // the sequential method is faster than thrust::inclusive_scan,
        // which uses high overhead dynamic parallel
        for (int i = begin + 1; i != end; i++)
        {
            precisely_summation_ref(sum, arclength_seg[i], residual);
            arclength_seg[i] = static_cast<Dt>(sum);
        }
    }
}

template <typename Dt, typename Curve, typename CurveLocator>
__global__ void kn_getAllArclength(Curve curve, Dt *arclength_seg, Dt *dst, CurveLocator locator)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (; tid < curve.n_curve; tid += blockDim.x * gridDim.x)
    {
        auto pos = locator(curve, tid);
        dst[tid] = arclength_seg[pos.end - 1];
    }
}

}