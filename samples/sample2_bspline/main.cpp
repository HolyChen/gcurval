#include <iostream>
#include <random>
#include <fstream>
#include <queue>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


#define USE_CUDA
#define USE_RMM
//#define ENABLE_NVVP

#define OUTPUT_RESULT

#include "arclength_calculate.h"
#include "dataset_loader.h"

using DataType = float;
using PointType = gcurval::Vector2X<DataType>;
const int degree = 3;

#ifdef USE_CUDA
using ArclengthCalculate = ArclengthCalculateGpu<DataType, PointType>;
#else
using ArclengthCalculate = ArclengthCalculateCpu<DataType, PointType>;
#endif // USE_CUDA

void do_test(const CurveSet<DataType, PointType>* cs, const int n_seg, const int n_sub, const int n_ptrs_per_curve)
{
    const int n_curve = cs->n_curve;

    ArclengthCalculate calculator(n_curve, cs->knots_offset, cs->knot_vector, cs->ctrl_ptrs);

    {
        gcurval::TimeCounter<> t;

        calculator.createArclengthTable();
        printf("%s\n", t.tell("ms", "Arc-Length Calculation:").c_str());
    }

#ifdef OUTPUT_RESULT
    {
        auto arc = calculator.getAllCurveArclengths();
        for (int i = 0; i < 10; i++)
        {
            printf("%d %f\n", i, arc[i]);
        }
    }
#endif // OUTPUT_RESULT

    int *sample_ptrs = new int[cs->n_curve];
    DataType *position = new DataType[n_ptrs_per_curve * cs->n_curve];
    for (int i = 0; i < n_curve; i++)
    {
        sample_ptrs[i] = n_ptrs_per_curve;
        for (int j = 0; j < n_ptrs_per_curve; j++)
        {
            position[i * n_ptrs_per_curve + j] = j * (1.0 / (n_ptrs_per_curve - 1));
        }
    }

    gcurval::ArclengthReverseResult<DataType, PointType> result;
    {
        gcurval::TimeCounter<> t;
#ifdef OUTPUT_RESULT
        result = calculator.arclengthReverse(sample_ptrs, position, true, gcurval::gcurvalHost,
            gcurval::gcurvalReverseReturnPoints);
#else
        result = calculator.arclengthReverse(sample_ptrs, position, true, gcurval::gcurvalDevice,
            gcurval::gcurvalReverseReturnUs);
#endif // OUTPUT_RESULT
        printf("%s\n", t.tell("ms", "Arc-Length Sampling:").c_str());
    }

#ifdef OUTPUT_RESULT

    for (int j = 0; j < 10; j++)
    {
        printf("\n\n\n============= %d =============\n", j);

        int begin = cs->knots_offset[j] - (degree + 1) * j;
        int end = cs->knots_offset[j + 1] - (degree + 1) * (j + 1);

        printf("[");
        for (int i = 0; i < (end - begin); i++)
        {
            printf("(%f, %f), ", cs->ctrl_ptrs[begin + i][0],
                cs->ctrl_ptrs[begin + i][1]);
        }
        printf("]");
        
        printf("\n---------------------\n");

        printf("[");
        for (int i = cs->knots_offset[j]; i < cs->knots_offset[j + 1]; i++)
        {
            printf("%f, ", cs->knot_vector[i]);
        }
        printf("]");

        printf("\n---------------------\n");

        printf("[");
        for (int i = 0; i < n_ptrs_per_curve; i++)
        {
            printf("(%f, %f), ", result.points[j * n_ptrs_per_curve + i][0],
                result.points[j * n_ptrs_per_curve + i][1]);
        }
        printf("]\n");
    }

#endif // OUTPUT_RESULT

}


int main(int argc, char* argv[])
{
    printf("Type: %s\n", sizeof(DataType) == sizeof(float) ? "float" : "double");

    {
#ifdef USE_CUDA
        // output cuda version
        int cuda_runtime_version;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        printf("CUDA Runtime: %d.%d\n", cuda_runtime_version / 1000, cuda_runtime_version % 1000);

        // set stack size for recursive
        cudaDeviceSetLimit(cudaLimitStackSize, 8192);

#ifdef ENABLE_NVVP
        printf("NVVP is enabled\n");
        cudaProfilerStart();
#endif // CUDA_PROFILE

#ifdef USE_RMM
        printf("Rmm is used\n");

        rmmOptions_t rmmOption;
        rmmOption.allocation_mode = PoolAllocation;
        rmmOption.initial_pool_size = 0x4000'0000; // 1 GB
        rmmOption.enable_logging = false;
        rmmInitialize(&rmmOption);
#endif // USE_RMM
#endif // USE_CUDA
    }

    // Set up the dataset
    const int n_curve = argc > 1 ? std::stoi(argv[1]) : 1000;
    const int n_ptrs_per_curve = argc > 2 ? std::stoi(argv[2]) : 10000;

    const char* files[] = {
        "dataset/BSpline/Bundle_0.txt",
        "dataset/BSpline/Bundle_1.txt",
        "dataset/BSpline/Bundle_2.txt",
        "dataset/BSpline/Bundle_3.txt",
        "dataset/BSpline/Bundle_4.txt",
        "dataset/BSpline/Bundle_5.txt",
        "dataset/BSpline/Bundle_6.txt",
        "dataset/BSpline/Bundle_7.txt",
        "dataset/BSpline/Bundle_8.txt",
        "dataset/BSpline/Bundle_9.txt"
    };


    auto cs = load_dataset<DataType, PointType>(n_curve, files, 10);

    const int n_seg = cs->knots_offset[cs->n_curve] - (cs->degree * 2 + 1) * cs->n_curve;
    const int n_sub = n_seg * gcurval:: N_SUBDIVIDED;

    do_test(cs, n_seg, n_sub, n_ptrs_per_curve);
    delete cs;

    {
#ifdef USE_CUDA

#ifdef USE_RMM
        rmmFinalize();
#endif // USE_RMM


#ifdef ENABLE_NVVP
        cudaProfilerStop();
#endif // ENABLE_NVVP

#endif // USE_CUDA
    }

    return 0;
}



