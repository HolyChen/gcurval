#pragma once

#include "../../base.h"

#include <numeric>

namespace gcurval
{

template <typename Curve,
    typename CurveTraits,
    int max_recursive_depth = MaxRecursiveDepth<typename Curve::DataType>::value>
class ArclengthCalculateCpu
{
public:
    using Dt = typename Curve::DataType;
    using PointType = typename Curve::PointType;
    using CalculateFunction = typename CurveTraits::CalculateFunction;
    using ParaPackParser = typename CurveTraits::ParaPackParser;
    using ParaPackSaver = typename CurveTraits::ParaPackSaver;
    using SharedMemAllocator = typename CurveTraits::SharedMemAllocator;
    using CurveLocator = typename CurveTraits::CurveLocator;
    using GetPoint = typename CurveTraits::GetPoint;

public:
    // The max of recursive depth of bisection in arclength calculation.
    // See: MaxRecursiveDepth
    static constexpr int MAX_RECURSIVE_DEPTH = max_recursive_depth;

protected:
    Curve curve;
    int n_seg = 0;
    
    Dt **u_table = nullptr;
    Dt **arclength_table = nullptr;
    int **segid_table = nullptr;
    int *table_length = nullptr;

    Dt *arclength = nullptr;

public:
     explicit ArclengthCalculateCpu(int n_curve, int n_seg)
        : n_seg(n_seg)
    {
        curve.n_curve = n_curve;

        u_table = new Dt*[n_curve];
        arclength_table = new Dt*[n_curve];
        segid_table = new int*[n_curve];
        table_length = new int[n_curve];
    }

    explicit ArclengthCalculateCpu(int n_curve)
        : n_seg(n_curve)
    {
        curve.n_curve = n_curve;

        u_table = new Dt*[n_curve];
        arclength_table = new Dt*[n_curve];
        segid_table = new int*[n_curve];
        table_length = new int[n_curve];
    }

    virtual ~ArclengthCalculateCpu()
    {
        for (int i = 0; i < curve.n_curve; i++)
        {
            gcurval::freeArray_s(u_table[i]);
            gcurval::freeArray_s(arclength_table[i]);
            gcurval::freeArray_s(segid_table[i]);
        }
        gcurval::freeArray_s(u_table);
        gcurval::freeArray_s(arclength_table);
        gcurval::freeArray_s(segid_table);
        gcurval::freeArray_s(table_length);

        gcurval::freeArray_s(arclength);
    }

    void createArclengthTable()
    {
        ParaPackParser parser;
        CurveLocator locator;
        CalculateFunction calculateFunction;
        DoArclengthCalculateStep<SharedMemAllocator> calculateStep;

        int init_buffer_size = ((n_seg - 1) / curve.n_curve + 1) * (8 * MAX_RECURSIVE_DEPTH);  // experiential
        std::vector<Dt> u_table_cid;
        std::vector<Dt> al_table_cid;
        std::vector<int> segid_table_cid;
        u_table_cid.reserve(init_buffer_size);
        al_table_cid.reserve(init_buffer_size);
        segid_table_cid.reserve(init_buffer_size);

        uint8_t* buffer = new uint8_t[std::max<int>(SharedMemAllocator::size(1), 1)]; // only a single thread
        
        for (int cid = 0; cid < curve.n_curve; cid++)
        {
            u_table_cid.clear();
            al_table_cid.clear();
            segid_table_cid.clear();

            auto interval = locator(curve, cid);
            auto n_seg_for_cid = interval.end - interval.begin;

            wider_real_type_t<Dt> al_acc_curve{}, residual{}; // accumulate arclength for this curve

            for (int segid = 0; segid < n_seg_for_cid; segid++)
            {
                auto pack = parser(curve, cid, segid, true);

                wider_real_type_t<Dt> al_acc_seg = {};
                Dt parent_arclength = calculateStep(calculateFunction, pack.begin, pack.end, pack, buffer);
                auto seg_store_begin = al_table_cid.size();
                al_acc_seg = arclengthRecursive(pack.begin, pack.end, parent_arclength, 0,
                    pack, buffer, al_acc_seg, segid, u_table_cid, al_table_cid, segid_table_cid);
                addToTable(pack.end, al_acc_seg, segid, u_table_cid, al_table_cid, segid_table_cid);
                auto seg_store_end = al_table_cid.size();

                std::for_each(al_table_cid.begin() + seg_store_begin,
                    al_table_cid.begin() + seg_store_end,
                    [al_acc_curve, residual](Dt &v) { v = Dt(precisely_summation(al_acc_curve, v, residual)); });
                precisely_summation_ref(al_acc_curve, al_acc_seg, residual);
            }

            table_length[cid] = u_table_cid.size();

            u_table[cid] = new Dt[u_table_cid.size()];
            std::copy(u_table_cid.begin(), u_table_cid.end(), u_table[cid]);

            arclength_table[cid] = new Dt[al_table_cid.size()];
            std::copy(al_table_cid.begin(), al_table_cid.end(), arclength_table[cid]);

            segid_table[cid] = new int[segid_table_cid.size()];
            std::copy(segid_table_cid.begin(), segid_table_cid.end(), segid_table[cid]);
        }

        delete[] buffer;
    }

    Dt* getAllCurveArclengths()
    {
        return getAllCurveArclengths_Impl(true);
    }

    Dt getArclength(int cid)
    {
        getAllCurveArclengths_Impl(false);
        return arclength[cid];
    }

    ArclengthReverseResult<Dt, PointType>
        arclengthReverse(int *n_ptrs_for_curves, Dt *sample_position, bool is_relative,
            GcurvalEnum result_is_device = gcurvalHost,
            GcurvalEnum result_u_or_point = gcurvalReverseReturnUs)
    {
        ParaPackParser parser;
        CurveLocator locator;
        CalculateFunction calculateFunction;
        DoArclengthCalculateStep<SharedMemAllocator> calculateStep;
        DoGetPointStep<SharedMemAllocator> getPointStep;
        GetPoint getPoint;
        
        uint8_t* buffer = new uint8_t[std::max<int>(SharedMemAllocator::size(1), 1)]; // only a single thread

        getAllCurveArclengths_Impl(false);

        int total_ptrs = 0;
        int *n_ptrs_acc = new int[curve.n_curve + 1];
        n_ptrs_acc[0] = 0;
        // do it on CPU is faster than on GPU with kernel launch overhead
        std::partial_sum(n_ptrs_for_curves, n_ptrs_for_curves + curve.n_curve, n_ptrs_acc + 1);
        total_ptrs = n_ptrs_acc[curve.n_curve];
        
        ArclengthReverseResult<Dt, PointType> result;
        result.n_points = total_ptrs;
        result.store_where = gcurvalHost;
        result.store_what = result_u_or_point;

        if (result_u_or_point == gcurvalReverseReturnUs)
        {
            result.us = new Dt[total_ptrs];
        }
        else
        {
            result.points = new PointType[total_ptrs];
        }

        for (int cid = 0; cid < curve.n_curve; cid++)
        {
            Dt al_cid = arclength[cid];

            for (int i_ptr = n_ptrs_acc[cid]; i_ptr < n_ptrs_acc[cid + 1]; i_ptr++)
            {
                Dt target = is_relative ? sample_position[i_ptr] * al_cid : sample_position[i_ptr];
                auto i_table = std::lower_bound(arclength_table[cid], arclength_table[cid] + table_length[cid],
                    target) - arclength_table[cid];
                i_table = (i_table == table_length[cid] ? table_length[cid] - 1 : i_table);

                Dt begin = 0.0, end = 1.0;
                int segid = 0;

                if (i_table > 0)
                {
                    target -= arclength_table[cid][i_table - 1];
                    begin = u_table[cid][i_table - 1];
                    segid = segid_table[cid][i_table];
                }

                end = u_table[cid][i_table];

                auto pack = parser(curve, cid, segid);

                Dt parent_arclength = calculateStep(calculateFunction, begin, end, pack, buffer);

                Dt result_u = end;
                bool found = false;
                /*arclengthRecursiveWithTarget
                    (begin, end, parent_arclength, target, 0, result_u, found, pack, buffer);*/

                arclengthRecursiveWithTarget_Paper
                    (begin, end, parent_arclength, target, 0.0, 0.0, 0, result_u, found, pack, buffer);
                
                if (result_u_or_point == gcurvalReverseReturnUs)
                {
                    result.us[i_ptr] = result_u;
                }
                else
                {

                    result.points[i_ptr] = getPointStep(getPoint, result_u, pack, buffer);
                }
            }
        }

        delete[] n_ptrs_acc;
        return result;
    }


protected:
    Dt arclengthRecursive(Dt begin, Dt end, Dt parent_arclength,
        int rec_level,
        const typename CurveTraits::ParaPack& pack,
        uint8_t *buffer,
        wider_real_type_t<Dt> total_length,
        int segid,
        std::vector<Dt>& u_table,
        std::vector<Dt>& al_table,
        std::vector<int>& segid_table)
    {
        using EPS = Epsilon<Dt>;

        CalculateFunction calculate_function;
        DoArclengthCalculateStep<SharedMemAllocator> do_step;

        Dt mid = (begin + end) / 2.0;

        Dt left_arclength = do_step(calculate_function, begin, mid, pack, buffer);
        Dt right_arclength = do_step(calculate_function, mid, end, pack, buffer);

        if (rec_level < MAX_RECURSIVE_DEPTH &&
            std::abs(left_arclength + right_arclength - parent_arclength) > EPS::v(rec_level))
        {
            left_arclength = arclengthRecursive(
                begin, mid, left_arclength, rec_level + 1, pack, buffer,
                total_length, segid, u_table, al_table, segid_table);

            total_length += left_arclength;
            addToTable(mid, (Dt)total_length, segid, u_table, al_table, segid_table);

            right_arclength = arclengthRecursive(
                mid, end, right_arclength, rec_level + 1, pack, buffer,
                total_length, segid, u_table, al_table, segid_table);
        }

        return left_arclength + right_arclength;
    }

    void addToTable(Dt u, Dt arclength, int segid,
        std::vector<Dt>& us, std::vector<Dt>& als, std::vector<int>& segs)
    {
        us.emplace_back(u);
        als.emplace_back(arclength);
        segs.emplace_back(segid);
    }

    Dt* getAllCurveArclengths_Impl(bool need_return = true)
    {
        if (arclength == nullptr)
        {
            arclength = new Dt[curve.n_curve];
            for (int cid = 0; cid < curve.n_curve; cid++)
            {
                arclength[cid] = arclength_table[cid][table_length[cid] - 1];
            }
        }

        Dt *result = nullptr;
        if (need_return)
        {
            result = new Dt[curve.n_curve];
            std::copy(arclength, arclength + curve.n_curve, result);
        }

        return result;
    }

    Dt arclengthRecursiveWithTarget(
        Dt begin, Dt end, Dt parent_arclength,
        Dt target,
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

        Dt left_right_sum = left_arclength + right_arclength;

        if (std::abs(begin - end) < EPS::REVERSE || rec_level == MAX_RECURSIVE_DEPTH)
        {
            if (target <= left_arclength)
            {
                u = (begin + mid) / 2.0;
                found = true;
            }
            else if (target <= (left_arclength + right_arclength))
            {
                u = (mid + end) / 2.0;
                found = true;
            }
        }
        else
        {
            if (std::abs(left_right_sum - parent_arclength) > EPS::v(rec_level))
            {
                left_arclength = arclengthRecursiveWithTarget
                    (begin, mid, left_arclength, target, rec_level + 1, u, found, pack, buffers);

                if (!found)
                {
                    right_arclength =
                        arclengthRecursiveWithTarget
                        (mid, end, right_arclength, target - left_arclength, rec_level + 1, u, found, pack, buffers);
                }
            }
            else
            {
                if (target < left_arclength)
                {
                    left_arclength =
                        left_arclength = arclengthRecursiveWithTarget
                        (begin, mid, left_arclength, target, rec_level + 1, u, found, pack, buffers);
                }

                if (!found && target <= left_right_sum)
                {
                    right_arclength =
                        arclengthRecursiveWithTarget
                        (mid, end, right_arclength, target - left_arclength, rec_level + 1, u, found, pack, buffers);
                }
            }
        }

        return left_right_sum;
    }

    Dt arclengthRecursiveWithTarget_Paper(
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
            left_arclength = arclengthRecursiveWithTarget_Paper(begin, mid, left_arclength,
                target, acc_arclength, residual, rec_level + 1, u, found, pack, buffers);
            if (!found)
            {
                precisely_summation_ref(acc_arclength, left_arclength, residual);
                right_arclength = arclengthRecursiveWithTarget_Paper(mid, end, right_arclength,
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
                left_arclength = arclengthRecursiveWithTarget_Paper(begin, mid, left_arclength,
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
                right_arclength = arclengthRecursiveWithTarget_Paper(mid, end, right_arclength,
                    target, acc_arclength, residual, rec_level + 1, u, found, pack, buffers);
            }
        }
        return left_arclength + right_arclength;
    }
};

}