#pragma once

#include "../util.h"
#include "../../../base.h"

namespace gcurval
{
// Arc-length calculation, naive method
template <typename Dt,
    typename Curve, typename CurveTraits,
    int MAX_RECURSIVE_DEPTH = MaxRecursiveDepth<Dt>::value>
GCURVAL_HD_FUNC Dt arclengthRecursive(Dt begin, Dt end, Dt parent_arclength, int rec_level,
        const typename CurveTraits::ParaPack& pack,
        uint8_t *buffer)
{
    using EPS = Epsilon<Dt>;

    typename CurveTraits::CalculateFunction calculate_function;
    DoArclengthCalculateStep<typename CurveTraits::SharedMemAllocator> do_step;

    Dt mid = (begin + end) / 2.0;

    Dt left_arclength = do_step(calculate_function, begin, mid, pack, buffer);
    Dt right_arclength = do_step(calculate_function, mid, end, pack, buffer);

    if (rec_level < MAX_RECURSIVE_DEPTH &&
        gcurval::abs(left_arclength + right_arclength - parent_arclength) > EPS::v(rec_level))
    {
        left_arclength = arclengthRecursive<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>(
            begin, mid, left_arclength, rec_level + 1, pack, buffer);
        right_arclength = arclengthRecursive<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>(
            mid, end, right_arclength, rec_level + 1, pack, buffer);
    }

    return left_arclength + right_arclength;
}

template <typename Dt,
    typename Curve,
    typename CurveTraits,
    int MAX_RECURSIVE_DEPTH = MaxRecursiveDepth<Dt>::value>
__global__ void kn_curveArclengthNaive(Curve curves,
        int n_task,
        Dt* result,
        typename CurveTraits::ForwardSpace fs,
        typename CurveTraits::BackwardSpace bs)
{

    typename CurveTraits::CalculateFunction calculate_function;
    DoArclengthCalculateStep<typename CurveTraits::SharedMemAllocator> do_step;
    typename CurveTraits::ParaPackParser paraPackParser;

    extern __shared__ uint8_t buffers[];

    auto tid = threadIdx.x + blockDim.x * blockIdx.x;

    Dt parent_arclength;
    Dt begin, end;

    if (tid < n_task)
    {
        auto pack = paraPackParser(curves, fs.cid[tid], fs.sid[tid]);
        begin = fs.begin[tid];
        end = fs.end[tid];
        parent_arclength = do_step(calculate_function, begin, end, pack, buffers);

        result[tid] = arclengthRecursive<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>(
            begin, end, parent_arclength, 0, pack, buffers);
    }
}


// Arc-length sampling naive method
template <typename Dt,
    typename CurveTraits,
    int MAX_RECURSIVE_DEPTH = MaxRecursiveDepth<Dt>::value>
    GCURVAL_HD_FUNC Dt arclengthRecursiveWithTarget_Paper(
        Dt begin, Dt end, Dt parent_arclength,
        Dt target,
        wider_real_type_t<Dt> acc_arclength,
        wider_real_type_t<Dt> residual,
        int rec_level,
        Dt& u,
        bool &found,
        const typename CurveTraits::ParaPack& pack,
        uint8_t* buffers)
{
    using EPS = Epsilon<Dt>;

    typename CurveTraits::CalculateFunction calculateFunc;
    DoArclengthCalculateStep<typename CurveTraits::SharedMemAllocator> calculateStep{};

    Dt mid = (begin + end) / 2.0;

    Dt left_arclength = calculateStep(calculateFunc, begin, mid, pack, buffers);
    Dt right_arclength = calculateStep(calculateFunc, mid, end, pack, buffers);

    if (rec_level < MAX_RECURSIVE_DEPTH &&
        std::abs((left_arclength + right_arclength) - parent_arclength) > EPS::v(rec_level))
    {
        left_arclength = arclengthRecursiveWithTarget_Paper
            <Dt, CurveTraits, MAX_RECURSIVE_DEPTH>
            (begin, mid, left_arclength,
                target, acc_arclength, residual, rec_level + 1, u, found, pack, buffers);
        if (!found)
        {
            precisely_summation_ref(acc_arclength, left_arclength, residual);
            right_arclength = arclengthRecursiveWithTarget_Paper
                <Dt, CurveTraits, MAX_RECURSIVE_DEPTH>
                (mid, end, right_arclength,
                    target, acc_arclength, residual, rec_level + 1, u, found, pack, buffers);
        }
    }
    else if (target <= precisely_summation(acc_arclength, left_arclength, residual))
    {
        if (mid - begin < EPS::REVERSE || rec_level == MAX_RECURSIVE_DEPTH)
        {
            u = (begin + mid) / Dt(2.0);
            found = true;
        }
        else
        {
            left_arclength = arclengthRecursiveWithTarget_Paper
                <Dt, CurveTraits, MAX_RECURSIVE_DEPTH>
                (begin, mid, left_arclength,
                    target, acc_arclength, residual, rec_level + 1, u, found, pack, buffers);
        }
    }
    else if (target <= precisely_summation(acc_arclength, left_arclength + right_arclength, residual))
    {
        if (end - mid < EPS::REVERSE || rec_level == MAX_RECURSIVE_DEPTH)
        {
            u = (mid + end) / Dt(2.0);
            found = true;
        }
        else
        {
            precisely_summation_ref(acc_arclength, left_arclength, residual);
            right_arclength = arclengthRecursiveWithTarget_Paper
                <Dt, CurveTraits, MAX_RECURSIVE_DEPTH>
                (mid, end, right_arclength,
                    target, acc_arclength, residual, rec_level + 1, u, found, pack, buffers);
        }
    }

    return left_arclength + right_arclength;
}



template <
    typename Dt,
    typename Curve,
    typename CurveTraits,
    int MAX_RECURSIVE_DEPTH = MaxRecursiveDepth<Dt>::value>
    __global__ void kn_arclengthReverseDfs(int n_ptrs, Curve curve,
        typename CurveTraits::ForwardSpace fs, typename CurveTraits::BackwardSpace bs)
{
    using EPS = Epsilon<Dt>;

    extern __shared__ uint8_t buffers[];

    auto tid = threadIdx.x + blockDim.x * blockIdx.x;

    int rec_level = 0;

    typename CurveTraits::CalculateFunction calculateFunction{};
    DoArclengthCalculateStep<typename CurveTraits::SharedMemAllocator> calculateStep{};

    Dt left_arclength, right_arclength, parent_arclength;
    int cid;
    Dt begin, end, mid;
    Dt target;

    if (tid < n_ptrs)
    {
        auto pack = typename CurveTraits::ParaPackParser{}(curve, fs.cid[tid], fs.sid[tid]);

        begin = fs.begin[tid];
        end = fs.end[tid];
        target = bs.pal[tid];

        parent_arclength = calculateStep(calculateFunction, begin, end, pack, buffers);

        Dt result_u = end;
        bool found = false;

        arclengthRecursiveWithTarget_Paper
            <Dt, CurveTraits, MAX_RECURSIVE_DEPTH>
            (begin, end, parent_arclength, target, 0.0, 0.0, 0, result_u, found, pack, buffers);

        fs.begin[tid] = result_u;
    }
}

}
