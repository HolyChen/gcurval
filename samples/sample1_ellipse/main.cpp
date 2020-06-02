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

#include "ellipse.h"

using DataType = float;

#ifdef USE_CUDA
using ArclengthCalculate = ArclengthCalculateGpu<DataType>;
#else
using ArclengthCalculate = ArclengthCalculateCpu<DataType>;
#endif // USE_CUDA

void make_data_set(const int n_curve, const int seed, DataType* as, DataType* bs)
{
    std::default_random_engine eg(seed);
    std::uniform_real_distribution<DataType> dis(0.1, 2.0);

    for (int i = 0; i < n_curve; i++)
    {
        auto a = DataType(1.0);
        auto b = dis(eg) * a;
        auto m = std::max(a, b);

        as[i] = a / m;
        bs[i] = b / m;
    }

    as[0] = DataType(1.0);
    bs[0] = DataType(1.0);
}

void do_test(const int n_curve, const int n_seg, const int n_sub, const int n_ptrs_per_curve, const int seed = 413)
{
    DataType *as = new DataType[n_curve], *bs = new DataType[n_curve];
    make_data_set(n_curve, seed, as, bs);

    ArclengthCalculate calculator(n_curve, as, bs);

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

    int *sample_ptrs = new int[n_curve];
    DataType *position = new DataType[n_ptrs_per_curve * n_curve];
    for (int i = 0; i < n_curve; i++)
    {
        sample_ptrs[i] = n_ptrs_per_curve;
        for (int j = 0; j < n_ptrs_per_curve; j++)
        {
            position[i * n_ptrs_per_curve + j] = j * (1.0 / n_ptrs_per_curve);
        }
    }


    RESULT_OF_T<decltype(&ArclengthCalculate::arclengthReverse)(
        ArclengthCalculate,
        int*, DataType*, bool, gcurval::GcurvalEnum, gcurval::GcurvalEnum)> result;
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

        printf("a,b: %f,%f", as[j], bs[j]);

        printf("\n---------------------\n");

        printf("[");
        for (int i = 0; i < n_ptrs_per_curve; i++)
        {
            //printf("%f, ", result.us[j * n_ptrs_per_curve + i]);
            printf("(%f, %f), ", result.points[j * n_ptrs_per_curve + i][0],
                result.points[j * n_ptrs_per_curve + i][1]);
        }
        printf("]\n");
    }
#endif // OUTPUT_RESULT

    delete[] as;
    delete[] bs;
}


int main(int argc, char* argv[])
{
#ifdef OUTPUT_RESULT

    printf("Warning: Output result mode is enabled, the performance could be affected!");

#endif // OUTPUT_RESULT

    printf("%s\n", sizeof(DataType) == sizeof(float) ? "float" : "double");

    {
#ifdef USE_CUDA
        // output cuda version
        int cuda_runtime_version;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        printf("CUDA Runtime: %d.%d\n", cuda_runtime_version / 1000, cuda_runtime_version % 1000);

        // set stack size for recursive
        cudaDeviceSetLimit(cudaLimitStackSize, 8192);

#ifdef ENABLE_NVVP
        printf("Warning: NVVP is enabled\n");
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
    const int seed = 413; // The room number of our laboratory.
    const int n_curve = argc > 1 ? std::stoi(argv[1]) : 1000;
    const int n_seg = n_curve;
    const int n_sub = n_seg * gcurval::N_SUBDIVIDED;
    const int n_ptrs_per_curve = argc > 2 ? std::stoi(argv[2]) : 10000;

    do_test(n_curve, n_seg, n_sub, n_ptrs_per_curve, seed);

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