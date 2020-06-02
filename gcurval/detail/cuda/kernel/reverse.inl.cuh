#pragma once

#include "../util.h"
#include "../../../base.h"

namespace gcurval
{

template <typename T>
__device__ inline const T* lower_bound(const T* first, const T* last, const T& val)
{
    auto count = last - first;

    while (0 < count)
    {	// divide and conquer, find half that contains answer
        auto count2 = count >> 1;
        const auto mid = first + count2;
        if (*mid < val)
        {	// try top half
            first = mid + 1;
            count -= count2 + 1;
        }
        else
        {
            count = count2;
        }
    }

    return first;
}

template <
    typename Dt,
    typename Curve,
    typename CurveTraits,
    int n_subdivided>
__global__ void kn_initSearchTask(
        Curve curve, int *n_ptrs_acc, const Dt* arclength_seg, const Dt* arclength_subdivided,
        typename CurveTraits::ForwardSpace fs, typename CurveTraits::BackwardSpace bs, bool is_relative = true)
{
    // a block tackle a curve, the curve id is blockIdx.x;
    const int cid = blockIdx.x;

    __shared__ int store_start;
    __shared__ int n_ptrs;
    __shared__ CurveIntervalPos curve_pos;
    __shared__ Dt arclength;

    if (threadIdx.x == 0)
    {
        store_start = n_ptrs_acc[cid];
        n_ptrs = n_ptrs_acc[cid + 1] - store_start;
        curve_pos = typename CurveTraits::CurveLocator{}(curve, cid);
        arclength = arclength_seg[curve_pos.end - 1];
    }

    __syncthreads();

    for (auto i = threadIdx.x; i < n_ptrs; i += blockDim.x)
    {   
        Dt al = bs.pal[store_start + i];
        if (is_relative)
        {
            al *= arclength;
        }

        // search in segments first
        
        const Dt *seg_pos = lower_bound(
            arclength_seg + curve_pos.begin, arclength_seg + curve_pos.end, al);
        int segid = (seg_pos - arclength_seg) - curve_pos.begin;

        if (segid != 0)
        {
            al -= *(seg_pos - 1);
        }

        int sub_offset = (curve_pos.begin + segid) * n_subdivided;

        const Dt *sub_pos = lower_bound(arclength_subdivided + sub_offset,
            arclength_subdivided + sub_offset + n_subdivided, al);
        int subid = (sub_pos - arclength_subdivided) - sub_offset;

        if (subid != 0)
        {
            al -= *(sub_pos - 1);
        }
        
        auto pack = typename CurveTraits::ParaPackParser{}(curve, cid, segid, true); // need interval

        Dt delta = (pack.end - pack.begin) / n_subdivided;
        Dt begin = pack.begin + delta * subid;
        Dt end = pack.begin + delta * (subid + 1);

        typename CurveTraits::ParaPackSaver{}(store_start + i, fs, bs, begin, end, 0, al, pack);
        __syncthreads();
    }
    
}

template <
    typename Dt,
    typename Curve,
    typename CurveTraits,
    int MAX_RECURSIVE_DEPTH = MaxRecursiveDepth<Dt>::value>
    __global__ void kn_arclengthReverseDfsFast_Paper_WithoutStack(int n_ptrs, Curve curve,
        typename CurveTraits::ForwardSpace fs, typename CurveTraits::BackwardSpace bs)
{
    using EPS = Epsilon<Dt>;

    extern __shared__ uint8_t buffers[];

    auto tid = threadIdx.x + blockDim.x * blockIdx.x;

    int rec_level = 0;

    typename CurveTraits::CalculateFunction calculateFunction{};
    DoArclengthCalculateStep<typename CurveTraits::SharedMemAllocator> calculateStep{};


    wider_real_type_t<Dt> acc_arclength = 0.0, residual = 0.0;

    Dt left_arclength, right_arclength, parent_arclength, left_right_sum;
    wider_real_type_t<Dt> begin, end, mid;

    Dt target;
    Dt result_u;

    if (tid < n_ptrs)
    {
        auto pack = typename CurveTraits::ParaPackParser{}(curve, fs.cid[tid], fs.sid[tid]);

        begin = fs.begin[tid];
        end = fs.end[tid];
        mid = (begin + end) / 2;
        target = bs.pal[tid];

        int path_visited = 0x0001;

        // reverse
        result_u = end;
        bool found = false;

        while (!found && path_visited != 0)
        {
            mid = (begin + end) / 2.0;

            parent_arclength = calculateStep(calculateFunction, begin, end, pack, buffers);
            left_arclength = calculateStep(calculateFunction, begin, mid, pack, buffers);
            right_arclength = calculateStep(calculateFunction, mid, end, pack, buffers);
            left_right_sum = left_arclength + right_arclength;

            if (rec_level < MAX_RECURSIVE_DEPTH &&
                gcurval::abs(left_right_sum - parent_arclength) > EPS::v(rec_level))
            {
                rec_level++;
                end = mid;
                path_visited <<= 1;
            }
            else if (target <= precisely_summation(acc_arclength, left_arclength, residual))
            {
                if (mid - begin < EPS::REVERSE || rec_level == MAX_RECURSIVE_DEPTH)
                {
                    found = true;
                    result_u = (begin + mid) / Dt(2.0);
                }
                else
                {
                    rec_level++;
                    end = mid;
                    path_visited <<= 1;
                }
            }
            else if (target <= precisely_summation(acc_arclength, left_right_sum, residual))
            {
                if (end - mid < EPS::REVERSE || rec_level == MAX_RECURSIVE_DEPTH)
                {
                    found = true;
                    result_u = (mid + end) / Dt(2.0);
                }
                else
                {
                    rec_level++;
                    begin = mid;
                    precisely_summation_ref(acc_arclength, left_arclength, residual);
                    path_visited = (path_visited << 1) | 0x01;
                }
            }
            else
            {
                int mul = path_visited - (path_visited & (path_visited + 1));
                int delta_r = __popc(mul);
                mul++;
                rec_level -= delta_r;
                path_visited >>= delta_r;
                if (path_visited != 0)
                {
                    auto tmp = end;
                    end += (end - begin) * mul;
                    begin = tmp;
                    precisely_summation_ref(acc_arclength, left_right_sum, residual);
                    path_visited |= 0x01;
                }
            }
        }
    }

    __syncthreads();

    if (tid < n_ptrs)
    {
        fs.begin[tid] = result_u;
    }
}

template <
    typename Dt,
    typename Curve,
    typename CurveTraits,
    int MAX_RECURSIVE_DEPTH = MaxRecursiveDepth<Dt>::value>
    __global__ void kn_arclengthReverseDfsFast_Paper(int n_ptrs, Curve curve,
        typename CurveTraits::ForwardSpace fs, typename CurveTraits::BackwardSpace bs)
{
    using EPS = Epsilon<Dt>;

    extern __shared__ uint8_t buffers[];

    auto tid = threadIdx.x + blockDim.x * blockIdx.x;

    int rec_level = 0;

    typename CurveTraits::CalculateFunction calculateFunction{};
    DoArclengthCalculateStep<typename CurveTraits::SharedMemAllocator> calculateStep{};


    wider_real_type_t<Dt> acc_arclength = 0.0, residual = 0.0;

    Dt left_arclength, right_arclength, parent_arclength, left_right_sum;
    wider_real_type_t<Dt> begin, end, mid;

    Dt target;
    Dt result_u;

    Dt stack[MAX_RECURSIVE_DEPTH + 1];

    if (tid < n_ptrs)
    {
        auto pack = typename CurveTraits::ParaPackParser{}(curve, fs.cid[tid], fs.sid[tid]);

        begin = fs.begin[tid];
        end = fs.end[tid];
        mid = (begin + end) / 2;
        target = bs.pal[tid];

        int stack_top = 0;
        int path_visited = 0x0001;

        // reverse
        result_u = end;
        bool found = false;

        parent_arclength = calculateStep(calculateFunction, begin, end, pack, buffers);

        while (!found && path_visited != 0)
        {
            mid = (begin + end) / 2.0;

            left_arclength = calculateStep(calculateFunction, begin, mid, pack, buffers);
            right_arclength = calculateStep(calculateFunction, mid, end, pack, buffers);
            left_right_sum = left_arclength + right_arclength;

            if (rec_level < MAX_RECURSIVE_DEPTH &&
                gcurval::abs(left_right_sum - parent_arclength) > EPS::v(rec_level))
            {
                rec_level++;
                end = mid;
                path_visited <<= 1;
                stack[++stack_top] = right_arclength;
                parent_arclength = left_arclength;
            }
            else if (target <= precisely_summation(acc_arclength, left_arclength, residual))
            {
                if (mid - begin < EPS::REVERSE || rec_level == MAX_RECURSIVE_DEPTH)
                {
                    found = true;
                    result_u = (begin + mid) / Dt(2.0);
                }
                else
                {
                    rec_level++;
                    end = mid;
                    path_visited <<= 1;
                    stack[++stack_top] = right_arclength;
                    parent_arclength = left_arclength;
                }
            }
            else if (target <= precisely_summation(acc_arclength, left_right_sum, residual))
            {
                if (end - mid < EPS::REVERSE || rec_level == MAX_RECURSIVE_DEPTH)
                {
                    found = true;
                    result_u = (mid + end) / Dt(2.0);
                }
                else
                {
                    rec_level++;
                    begin = mid;
                    precisely_summation_ref(acc_arclength, left_arclength, residual);
                    path_visited = (path_visited << 1) | 0x01;
                    parent_arclength = right_arclength;
                }
            }
            else
            {
                int mul = path_visited - (path_visited & (path_visited + 1));
                int delta_r = __popc(mul);
                mul++;
                rec_level -= delta_r;
                path_visited >>= delta_r;
                if (path_visited != 0)
                {
                    auto tmp = end;
                    end += (end - begin) * mul;
                    begin = tmp;
                    precisely_summation_ref(acc_arclength, left_right_sum, residual);
                    path_visited |= 0x01;
                    parent_arclength = stack[stack_top];
                }

                stack_top--;
            }
        }
    }

    __syncthreads();

    if (tid < n_ptrs)
    {
        fs.begin[tid] = result_u;
    }
}


template <typename Dt, typename Curve, typename CurveTraits>
__global__ void kn_toPoints(int n_ptrs,
    Curve curve, typename CurveTraits::ForwardSpace fs,
    typename CurveTraits::PointType *ptrs)
{
    extern __shared__ uint8_t buffers[];

    auto tid = threadIdx.x + blockDim.x * blockIdx.x;

    typename CurveTraits::GetPoint getPoint{};
    DoGetPointStep<typename CurveTraits::SharedMemAllocator> doGetPointStep{};

    if (tid < n_ptrs)
    {
        auto pack = typename CurveTraits::ParaPackParser{}(curve, fs.cid[tid], fs.sid[tid]);
        auto u = fs.begin[tid];

        ptrs[tid] = doGetPointStep(getPoint, u, pack, buffers);
    }
}

}