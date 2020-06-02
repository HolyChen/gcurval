#pragma once

#include "../util.h"
#include "../../../base.h"

namespace gcurval
{

template <typename Dt,
    typename Curve,
    typename CurveTraits,
    int MAX_RECURSIVE_DEPTH = MaxRecursiveDepth<Dt>::value>
    __global__ void kn_curveArclengthForward(Curve curves,
        typename CurveTraits::ForwardSpace fsCur,
        typename CurveTraits::ForwardSpace fsNext,
        typename CurveTraits::BackwardSpace bsCur,
        typename CurveTraits::BackwardSpace bsNext,
        int n_task_cur, int *n_task_next,
        int rec_level)
{
    extern __shared__ uint8_t buffers[];

    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int n_sub_stask;

    if (threadIdx.x == 0)
    {
        n_sub_stask = 0;
    }

    __syncthreads();

    int store_pos_in_block = -1;

    Dt parent_arclength, left_arclength, right_arclength;
    Dt begin, end, mid;
    int cid;

    typename CurveTraits::ParaPack pack;
    typename CurveTraits::CalculateFunction calculateFunc;
    typename CurveTraits::ParaPackParser paraPackParser;
    typename CurveTraits::ParaPackSaver paraPackSaver;
    DoArclengthCalculateStep<typename CurveTraits::SharedMemAllocator> calculateStep;

    using EPS = Epsilon<Dt>;

    if (tid < n_task_cur)
    {
        pack = paraPackParser(curves, fsCur.cid[tid], fsCur.sid[tid]);

        begin = fsCur.begin[tid];
        end = fsCur.end[tid];
        cid = pack.cid;
        // omit <3> ptid
        parent_arclength = bsCur.pal[tid];

        if (rec_level == 0) // unset, the first level
        {
            parent_arclength = calculateStep(calculateFunc, begin, end, pack, buffers);
        }

        mid = (begin + end) / 2.0f;

        left_arclength = calculateStep(calculateFunc, begin, mid, pack, buffers);
        right_arclength = calculateStep(calculateFunc, mid, end, pack, buffers);

        // Subdivide the interval to 2 halfs
        if (rec_level < MAX_RECURSIVE_DEPTH && gcurval::abs(left_arclength + right_arclength - parent_arclength) > EPS::v(rec_level))
        {
            store_pos_in_block = atomicAdd(&n_sub_stask, 2);
        }
        else
        {
            bsCur.pal[tid] = left_arclength + right_arclength;
        }
    }
    __syncthreads();

    __shared__ int stask_q_start;

    if (rec_level < MAX_RECURSIVE_DEPTH && threadIdx.x == 0)
    {
        stask_q_start = atomicAdd(n_task_next, n_sub_stask);
    }

    __syncthreads();

    if (store_pos_in_block >= 0 && tid < n_task_cur)
    {
        int pos = stask_q_start + store_pos_in_block;

        paraPackSaver(pos, fsNext, bsNext, begin, mid, tid, left_arclength, pack);

        pos++;

        paraPackSaver(pos, fsNext, bsNext, mid, end, tid, right_arclength, pack);
    }
}

template <typename Dt>
__global__ void kn_curveArclengthBackward_shfl(
    Dt* seg_stask_parent_arclength_cur,
    Dt* seg_stask_parent_arclength_pre,
    int* seg_stask_parent_tid_cur,
    int n_seg,
    int rec_level)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;


    if (tid < n_seg)
    {
        Dt this_arclength = seg_stask_parent_arclength_cur[tid];
        int parent_id = seg_stask_parent_tid_cur[tid];

        // using warp shuffle to merge left and right half interval arclength
        this_arclength += __shfl_down_sync(0xFFFFFFFF, this_arclength, 1);

        // store the arclength back into parent task in last forward iteration
        if (!(tid & 0x1)) // even
        {
            seg_stask_parent_arclength_pre[parent_id] = this_arclength;
        }
    }
}

}
