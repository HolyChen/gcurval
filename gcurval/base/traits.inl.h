#pragma once

#include "util.h"

#include "structs.inl.h"
#include "functors.inl.h"

namespace gcurval
{

template <typename Curve>
struct CurveTraitsBase
{
    using DataType = typename Curve::DataType;
    using PointType = typename Curve::PointType;

    using ForwardSpace = ForwardSpaceBase<DataType>;
    using BackwardSpace = BackwardSpaceBase<DataType>;

    using ParaPackParser = ParaPackParserBase<DataType, PointType>;
    using ParaPackSaver = ParaPackSaverBase<DataType>;
    using ParaPack = ParaPackBase<DataType>;

    using SharedMemAllocator = NoSharedMemAllocator;
    using CurveLocator = CurveLocatorBase<Curve>;

    using GetPoint = void;
    using GetDerivative = void;
    using CalculateFunction = void;
};

}
