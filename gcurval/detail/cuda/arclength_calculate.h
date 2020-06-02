#pragma once

#include "util.h"
#include "../../base.h"
#include "kernel/_all.cuh"

#include <type_traits>
#include <numeric>

namespace gcurval
{

template <
    typename Dt,
    int max_recursive_depth,
    typename ForwardSpace = ForwardSpaceBase<Dt>,
    typename BackwardSpace = BackwardSpaceBase<Dt>>
struct BufferManagerBase
{
    static constexpr int N_FSPACE = 2;
    static constexpr int N_BSPACE = max_recursive_depth + 1;

public:
    ForwardSpace d_fspace[N_FSPACE];
    int fs_capacity[N_FSPACE] = { 0 };

    BackwardSpace d_bspace[N_BSPACE];
    int bs_capacity[N_BSPACE] = { 0 };

public:
    void fSpaceAllocate(const int which, int n_task)
    {
        assert(which >= 0 && which < N_FSPACE);
        gMalloc(&d_fspace[which].begin, sizeof(Dt) * n_task);
        gMalloc(&d_fspace[which].end, sizeof(Dt) * n_task);
        gMalloc(&d_fspace[which].cid, sizeof(int) * n_task);
        gMalloc(&d_fspace[which].sid, sizeof(int) * n_task);

        fs_capacity[which] = n_task;
    }

    void fSpaceFree(int which)
    {
        assert(which >= 0 && which < N_FSPACE);

        gFree_s(d_fspace[which].begin);
        gFree_s(d_fspace[which].end);
        gFree_s(d_fspace[which].cid);
        gFree_s(d_fspace[which].sid);

        fs_capacity[which] = 0;
    }

    void bSpaceAllocate(int which, int n_task)
    {
        assert(which >= 0 && which < N_BSPACE);

        gMalloc(&d_bspace[which].pal, sizeof(Dt) * n_task);
        gMalloc(&d_bspace[which].ptid, sizeof(int) * n_task);
        bs_capacity[which] = n_task;
    }

    void bSpaceFree(int which)
    {
        assert(which >= 0 && which <= N_BSPACE);

        gFree_s(d_bspace[which].pal);
        gFree_s(d_bspace[which].ptid);

        bs_capacity[which] = 0;
    }

    void fbSpacesFreeAll()
    {
        for (auto i = 0; i < N_FSPACE; i++) fSpaceFree(i);
        for (auto i = 0; i < N_BSPACE; i++) bSpaceFree(i);
    }

public:
    static constexpr int fsCurId(int rec)
    {
        return fsNextKId(rec, 0);
    }

    // Get index of forward space in next recusive
    static constexpr int fsNextKId(int rec, int k = 1)
    {
        return (rec + k) % N_FSPACE;
    }
};


template <typename Curve,
    typename CurveTraits,
    int max_recursive_depth = MaxRecursiveDepth<Curve::Dt>::value,
    int n_subdivided = gcurval::N_SUBDIVIDED, // 2 warp size 
    typename BufferManager =
        BufferManagerBase<typename Curve::DataType, max_recursive_depth,
            typename CurveTraits::ForwardSpace, typename CurveTraits::BackwardSpace>>
class ArclengthCalculateCuda
{
    // The 1024 is the max of blockDim.x in cuda device for CUDA 10.
    static_assert(n_subdivided <= 1024u, "The number of subdivided seg must be no more than 1024");

public:
    using Dt = typename Curve::DataType;
    using PointType = typename Curve::PointType;
    using CalculateFunction = typename CurveTraits::CalculateFunction;
    using ParaPackParser = typename CurveTraits::ParaPackParser;
    using ParaPackSaver = typename CurveTraits::ParaPackSaver;
    using SharedMemAllocator = typename CurveTraits::SharedMemAllocator;
    using CurveLocator = typename CurveTraits::CurveLocator;

public:
    // How many subsegment should be subdivided from a single segment.
    static constexpr int N_SUBDIVIDE = n_subdivided;
    // The max of recursive depth of bisection in arclength calculation.
    // See: MaxRecursiveDepth
    static constexpr int MAX_RECURSIVE_DEPTH = max_recursive_depth;

    static constexpr int N_FSPACE = 2;
    static constexpr int N_BSPACE = MAX_RECURSIVE_DEPTH + 1;

protected:
    int n_seg = 0;
    int n_sub = 0;

    Dt *d_arclength_seg = nullptr;
    Dt *d_arclength_subdivided = nullptr;
    Dt *d_arclength = nullptr;
    Dt *h_arclength = nullptr;


    // BFS assitants
    int *d_n_tasks = nullptr;
    int *n_tasks = nullptr;

    Curve d_curve;
    BufferManager bufMng;
public:
    explicit ArclengthCalculateCuda(int n_curve, int n_seg)
        : n_seg(n_seg)
    {
        d_curve.n_curve = n_curve;
        this->n_sub = n_seg * N_SUBDIVIDE;

        // d_n_task is used to store the number of task to be calculate and calculated,
        // the device write this memory while host read this memroy.
        
        cudaMallocHost(&n_tasks, sizeof(int) * (MAX_RECURSIVE_DEPTH + 1));
        gMalloc(&d_n_tasks, sizeof(int) * (MAX_RECURSIVE_DEPTH + 1));
        gMalloc(&d_arclength_seg, sizeof(Dt) * n_seg);
    }

    explicit ArclengthCalculateCuda(int n_curve)
        : n_seg(n_curve)
    {
        d_curve.n_curve = n_curve;
        this->n_sub = n_seg * N_SUBDIVIDE;

        // d_n_task is used to store the number of task to be calculate and calculated,
        // the device write this memory while host read this memroy.
        cudaMallocHost(&n_tasks, sizeof(int) * (MAX_RECURSIVE_DEPTH + 1));
        gMalloc(&d_n_tasks, sizeof(int) * (MAX_RECURSIVE_DEPTH + 1));
        gMalloc(&d_arclength_seg, sizeof(Dt) * n_seg);
    }

    virtual ~ArclengthCalculateCuda()
    {
        gFree_s(d_n_tasks);
        gFree_s(d_arclength_subdivided);
        gFree_s(d_arclength_seg);
        if (n_seg > d_curve.n_curve)
        {
            gFree_s(d_arclength);
        }
        else
        {
            d_arclength = nullptr;
        }
        if (n_tasks) { cudaFreeHost(n_tasks); n_tasks = nullptr; }
        freeArray_s(h_arclength);
    }

    void createArclengthTable()
    {
        bufMng.fSpaceAllocate(0, n_sub);
        bufMng.bSpaceAllocate(0, n_sub);

        initializeTasks();
        // the space to store seg result
        d_arclength_subdivided = bufMng.d_bspace[0].pal;
        cc(cudaDeviceSynchronize());
        cet();

        // To avoid skipping by user's code
        arclengthCalculateBfs(n_sub, d_arclength_subdivided);
        //arclengthCalculateNaive(n_sub, d_arclength_subdivided);

        mergeSubToSeg();

        // accumulate arclength in seg if in need where a curve is divided into multiple segment
        if (n_seg > d_curve.n_curve)
        {
            accumulateArclengthBySeg();
        }

        bufMng.d_bspace[0].pal = nullptr;
        bufMng.fbSpacesFreeAll();
    }

    Dt* getAllCurveArclengths()
    {
        assert(d_arclength_seg != nullptr); // arclength table should be constructed first
        return getAllCurveArclengths_Impl(true);
    }

    Dt getArclength(int cid)
    {
        getAllCurveArclengths_Impl(false);
        return h_arclength[cid];
    }

    ArclengthReverseResult<Dt, PointType>
    arclengthReverse(int *n_ptrs_for_curves, Dt *sample_position, bool is_relative,
        GcurvalEnum result_is_device = gcurvalDevice,
        GcurvalEnum result_u_or_point = gcurvalReverseReturnUs)
    {
        return arclengthReverseDfs(n_ptrs_for_curves, sample_position, is_relative,
            result_is_device, result_u_or_point);
    }

    void clearArclength()
    {
        freeArray_s(h_arclength);
        gFree_s(d_arclength);
    }

protected: // -------------------- implement ------------------

    // Initial arclength calculation task, user defined.
    void initializeTasks()
    {
        const int blockDim_x = N_SUBDIVIDE;
        kn_curveLengthTableInit<Curve, CurveTraits, N_SUBDIVIDE>
            <<<d_curve.n_curve, blockDim_x>>>
            (d_curve, bufMng.d_fspace[0], bufMng.d_bspace[0]);
    }

    void arclengthCalculateNaive(int n_task, Dt* result)
    {
        assert(n_task > 0 && result != nullptr);

        arclengthCalculateNaive_Impl(n_task, result);
        cc(cudaDeviceSynchronize());
        cet();
    }

    void arclengthCalculateBfs(int init_n_task, Dt* result)
    {
        assert(init_n_task > 0 && result != nullptr);

        // pass how many task to calculate to device
        cudaMemset(d_n_tasks, 0, sizeof(int) * (MAX_RECURSIVE_DEPTH + 1));
        cudaMemcpy(d_n_tasks, &init_n_task, sizeof(int), cudaMemcpyHostToDevice);

        int rec_level = 0;
        for (/* rec_level = 0 */; rec_level <= MAX_RECURSIVE_DEPTH; rec_level++)
        {
            int n_task_cur;
            cudaMemcpy(&n_tasks[rec_level], &d_n_tasks[rec_level], sizeof(int), cudaMemcpyDeviceToHost);
            n_task_cur = n_tasks[rec_level];

            if (n_task_cur == 0)
            {
                break;
            }

            // detect buffer is sufficient or not,
            // for latter one, reallocation will execute.
            if (rec_level < MAX_RECURSIVE_DEPTH)
            {
                // Forward Space
                {
                    int curId = bufMng.fsCurId(rec_level);
                    int nextId = bufMng.fsNextKId(rec_level);

                    if (n_task_cur * 2 > bufMng.fs_capacity[nextId])
                    {
                        bufMng.fSpaceFree(nextId);
                        bufMng.fSpaceAllocate(nextId, n_task_cur * 2); // at most double tasks
                    }
                }

                // Backward Space
                if (n_task_cur * 2 > bufMng.bs_capacity[rec_level + 1])
                {
                    bufMng.bSpaceFree(rec_level + 1);
                    bufMng.bSpaceAllocate(rec_level + 1, n_task_cur * 2); // at most double tasks
                }

            }

            calculateForward_Impl(n_task_cur, rec_level);
            cc(cudaDeviceSynchronize());
            cet();
        }

        int blockDim_x_backward = 128;
        int shared_memory_backward = sizeof(Dt) * blockDim_x_backward / 2;

        for (rec_level--; rec_level > 0; rec_level--)
        {
            int n_stask = n_tasks[rec_level];
            kn_curveArclengthBackward_shfl
            <<<(n_stask - 1) / blockDim_x_backward + 1, blockDim_x_backward>>>
            (bufMng.d_bspace[rec_level].pal, bufMng.d_bspace[rec_level - 1].pal,
                    bufMng.d_bspace[rec_level].ptid, n_stask, rec_level);

            cc(cudaDeviceSynchronize());
            cet();
        }
    }

    void arclengthCalculateNaive_Impl(int n_task, Dt* result, int = 0)
    {
        int blockDim_x = 128;

        kn_curveArclengthNaive<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>
            <<<(n_task - 1) / blockDim_x + 1, blockDim_x, SharedMemAllocator::size(blockDim_x)>>>
            (d_curve, n_task, result, bufMng.d_fspace[0], bufMng.d_bspace[0]);
    }

    void calculateForward_Impl(int n_task, int rec_level)
    {
        int blockDim_x_forward = 128;

        kn_curveArclengthForward<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>
            <<<(n_task - 1) / blockDim_x_forward + 1, blockDim_x_forward, SharedMemAllocator::size(blockDim_x_forward)>>>
            (d_curve,
                bufMng.d_fspace[bufMng.fsCurId(rec_level)],
                bufMng.d_fspace[bufMng.fsNextKId(rec_level)],
                bufMng.d_bspace[rec_level], bufMng.d_bspace[rec_level + 1],
                n_task, &d_n_tasks[rec_level + 1],
                rec_level);
    }

    void mergeSubToSeg()
    {
        // merge arclength of subdivided task into whole seg that contains it.
        // Brent-Kun Algorithm is in used.
        {
            int blockDim_x = N_SUBDIVIDE;
            int shared_memory_size = sizeof(Dt) * blockDim_x;
            kn_mergeSubdividedIntoSeg<Dt, N_SUBDIVIDE>
                <<<(n_sub - 1) / blockDim_x + 1, blockDim_x, shared_memory_size>>>
                (d_arclength_subdivided, d_arclength_seg, n_sub);
            cc(cudaDeviceSynchronize());
            cet();
        }
    }

    void accumulateArclengthBySeg()
    {
        const int blockDim_x = 64;
        kn_mergeSeg<<<(d_curve.n_curve - 1) / blockDim_x + 1, blockDim_x>>>
            (d_curve, d_arclength_seg, CurveLocator());
        cc(cudaDeviceSynchronize());
        cet();
    }

    Dt* getAllCurveArclengths_Impl(bool need_return = true)
    {
        if (d_arclength == nullptr)
        {
            h_arclength = new Dt[d_curve.n_curve];

            if (n_seg > d_curve.n_curve)
            {
                gMalloc(&d_arclength, sizeof(Dt) * (d_curve.n_curve));

                const int blockDim_x = 64;
                kn_getAllArclength
                    <<<(d_curve.n_curve - 1) / blockDim_x + 1, blockDim_x>>>
                    (d_curve, d_arclength_seg, d_arclength, CurveLocator());

                cc(cudaDeviceSynchronize());
                cet();
            }
            else
            {
                d_arclength = d_arclength_seg;
            }

            cudaMemcpy(h_arclength, d_arclength, sizeof(Dt) * d_curve.n_curve, cudaMemcpyDeviceToHost);
        }

        Dt *result = nullptr;
        if (need_return)
        {
            result = new Dt[d_curve.n_curve];
            std::copy(h_arclength, h_arclength + d_curve.n_curve, result);
        }

        return result;
    }

    ArclengthReverseResult<Dt, PointType>
    arclengthReverseDfs(int *n_ptrs_for_curves, Dt *sample_position, bool is_relative,
        GcurvalEnum result_is_device = gcurvalDevice,
        GcurvalEnum result_u_or_point = gcurvalReverseReturnUs)
    {
        // 1. calculate arc length to find        
        int *d_n_ptrs_acc = nullptr;
        gMalloc(&d_n_ptrs_acc, sizeof(int) * (d_curve.n_curve + 1));
        int total_ptrs = 0;

        {
            int *n_ptrs_acc = new int[d_curve.n_curve + 1];
            n_ptrs_acc[0] = 0;
            // do it on CPU is faster than on GPU with kernel launch overhead
            std::partial_sum(n_ptrs_for_curves, n_ptrs_for_curves + d_curve.n_curve, n_ptrs_acc + 1);
            total_ptrs = n_ptrs_acc[d_curve.n_curve];

            cc(cudaMemcpy(d_n_ptrs_acc, n_ptrs_acc, sizeof(int) * (d_curve.n_curve + 1), cudaMemcpyHostToDevice));
            delete[] n_ptrs_acc;
        }
        
        // use dfs search, so only a forward space to store begin, end and other information about search interval;
        // only a Dt[] to store the arclength to search.
        // For convinient, the backward is used which contains a Dt[], and we can reuse PackParser and PackSaver
        bufMng.fSpaceAllocate(0, total_ptrs);
        bufMng.bSpaceAllocate(0, total_ptrs);
        Dt *& arclength_to_search = bufMng.d_bspace[0].pal; // alias for bufMng.d_bspace[0].pal

        cc(cudaMemcpy(arclength_to_search, sample_position, sizeof(Dt) * total_ptrs, cudaMemcpyHostToDevice));

        int blockDim_x = 64;

        // the position is a percent position w.r.t. total arclength of the curve,
        // so, the absolute arclength should be calculate by (sample_position[i] * curve[w.r.t. i].curve_length)
        kn_initSearchTask<Dt, Curve, CurveTraits, n_subdivided>
            <<<d_curve.n_curve, blockDim_x>>>
        (d_curve, d_n_ptrs_acc, d_arclength_seg, d_arclength_subdivided,
            bufMng.d_fspace[0], bufMng.d_bspace[0], is_relative);
        cc(cudaDeviceSynchronize());
        cet();

        kn_arclengthReverseDfsFast_Paper<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>
        //kn_arclengthReverseDfsFast_Paper_WithoutStack<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>
        //kn_arclengthReverseDfs<Dt, Curve, CurveTraits, MAX_RECURSIVE_DEPTH>
            <<<(total_ptrs - 1) / blockDim_x + 1, blockDim_x, CurveTraits::SharedMemAllocator::size(blockDim_x)>>>
        (total_ptrs, d_curve, bufMng.d_fspace[0],
            bufMng.d_bspace[0]);
        cc(cudaDeviceSynchronize());
        cet();

        ArclengthReverseResult<Dt, PointType> result;
        result.n_points = total_ptrs;
        result.store_where = result_is_device;
        result.store_what = result_u_or_point;
        if (result_is_device == gcurvalDevice)
        {
            if (result_u_or_point == gcurvalReverseReturnUs)
            {
                std::swap(result.us, bufMng.d_fspace[0].begin);
            }
            else
            {
                gMalloc(&result.points, sizeof(PointType) * total_ptrs);
                kn_toPoints<Dt, Curve, CurveTraits>
                    <<<(total_ptrs - 1) / blockDim_x + 1, blockDim_x, CurveTraits::SharedMemAllocator::size(blockDim_x)>>>
                (total_ptrs, d_curve, bufMng.d_fspace[0], result.points);
                cc(cudaDeviceSynchronize());
                cet();
            }
        }
        else
        {
            if (result_u_or_point == gcurvalReverseReturnUs)
            {
                result.us = new Dt[total_ptrs];
                cudaMemcpy(result.us, bufMng.d_fspace[0].begin, sizeof(Dt) * total_ptrs,
                    cudaMemcpyDeviceToHost);
            }
            else
            {
                PointType *d_points;
                gMalloc(&d_points, sizeof(PointType) * total_ptrs);
                kn_toPoints<Dt, Curve, CurveTraits>
                    <<<(total_ptrs - 1) / blockDim_x + 1, blockDim_x, CurveTraits::SharedMemAllocator::size(blockDim_x)>>>
                    (total_ptrs, d_curve, bufMng.d_fspace[0], d_points);
                result.points = new PointType[total_ptrs];
                cc(cudaDeviceSynchronize());
                cet();
                cudaMemcpy(result.points, d_points, sizeof(PointType) * (total_ptrs), cudaMemcpyDeviceToHost);
                gFree_s(d_points);
            }
        }

        gFree_s(d_n_ptrs_acc);
        bufMng.fbSpacesFreeAll();

        return result;
    }

};

}
