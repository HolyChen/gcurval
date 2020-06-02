#pragma once

#include "util.h"

namespace gcurval
{

template <typename Dt>
struct ForwardSpaceBase
{
    Dt *begin = nullptr;
    Dt *end = nullptr;
    int *cid = nullptr; // curve id
    int *sid = nullptr; // segment id
};

template <typename Dt>
struct BackwardSpaceBase
{
    int *ptid = nullptr;
    Dt *pal = nullptr;
};

template <typename Dt, typename Pt>
struct CurvesBase
{
    using DataType = Dt;
    using PointType = Pt;

public:

    int n_curve = 0;

public:
    GCURVAL_HD_FUNC CurvesBase(int n_curve = 0)
        : n_curve(n_curve)
    {}
};

template <typename Dt>
struct ParaPackBase
{
    Dt begin;
    Dt end;
    int cid;
    int sid;
};

template <typename T = void>
struct CurveIntervalPos_
{
    int begin;
    int end;
};

using CurveIntervalPos = CurveIntervalPos_<void>;

struct NoSharedMemAllocator
{
    static constexpr int size(int)
    {
        return 0;
    }
};

template <typename DataType, typename PointType>
struct ArclengthReverseResult
{
    int n_points = 0;
    GcurvalEnum store_where;
    GcurvalEnum store_what;
    union
    {
        DataType *us = nullptr;
        PointType *points;
    };
};

}