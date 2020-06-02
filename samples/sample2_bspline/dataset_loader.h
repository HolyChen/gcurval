#ifndef __DATASET_LOADER_H__
#define __DATASET_LOADER_H__

#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include <gcurval/base.h>

template <typename Dt, typename PointType>
struct CurveSet
{
    using Vector = PointType;

    int degree = 0;
    int n_curve = 0;
    int *knots_offset = nullptr;
    Dt *knot_vector = nullptr;
    Vector *ctrl_ptrs = nullptr;

public:
    CurveSet(int n_curve)
        : n_curve(n_curve)
    {
        knot_vector = new Dt[n_curve * 50];
        knots_offset = new int[n_curve + 1];
        ctrl_ptrs = new Vector[n_curve * 50];
    }

    ~CurveSet()
    {
        delete[] knots_offset;
        delete[] knot_vector;
        delete[] ctrl_ptrs;
    }
};

template <typename Dt, typename PointType>
void read_a_curve(std::ifstream& fin,
    Dt *knot_vector, int &knot_offset, PointType *ctrl_ptrs)
{
    using Vector = PointType;

    int degree, n_ctrl_ptrs, n_us;

    fin >> degree >> n_ctrl_ptrs >> n_us;

    for (int i_ctrl_ptrs = 0; i_ctrl_ptrs < n_ctrl_ptrs; i_ctrl_ptrs++)
    {
        fin >> ctrl_ptrs[i_ctrl_ptrs][0]; // x
    }
    for (int i_ctrl_ptrs = 0; i_ctrl_ptrs < n_ctrl_ptrs; i_ctrl_ptrs++)
    {
        fin >> ctrl_ptrs[i_ctrl_ptrs][1]; // y
        if (PointType().rows() == 3)
        {
            ctrl_ptrs[i_ctrl_ptrs][2] = 0.0; // z;
        }
    }

    for (int i_us = 0; i_us < n_us; i_us++)
    {
        fin >> knot_vector[i_us];
    }

    knot_offset = n_us;

    // Zero centric
    Vector centric = Vector::Zero();

    for (int i = 0; i < n_ctrl_ptrs; i++)
    {
        centric += ctrl_ptrs[i];
    }

    centric /= Dt(n_ctrl_ptrs);

    Dt max_xyz = 0.0;

    for (int i = 0; i < n_ctrl_ptrs; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            ctrl_ptrs[i][j] -= centric[j];

            max_xyz = std::max(max_xyz, (Dt)std::abs(ctrl_ptrs[i][j]));
        }
    }

    // Normalize
    for (int i = 0; i < n_ctrl_ptrs; i++)
    {
        ctrl_ptrs[i] /= max_xyz;
    }

}

template <typename Dt, typename PointType>
CurveSet<Dt, PointType>* load_dataset(int n_curve, const char* input_file_name[], int n_files)
{
    CurveSet<Dt, PointType> *cs = new CurveSet<Dt, PointType>(n_curve);
    cs->degree = 3;
    cs->knots_offset[0] = 0;

    int i_curve = 0;

    for (int i_file = 0; i_file * 1000 < n_curve; i_file++)
    {
        std::ifstream fin(input_file_name[i_file]);

        std::uniform_real_distribution<Dt> us_dis(1.0f, 3.0f);

        int end_curve = i_curve + std::min(1000, n_curve - i_file * 1000);
        for (; i_curve < end_curve; i_curve++)
        {
            int n_knots = 0;
            read_a_curve<Dt, PointType>(fin, &cs->knot_vector[cs->knots_offset[i_curve]], n_knots,
                &cs->ctrl_ptrs[cs->knots_offset[i_curve] - (cs->degree + 1) * i_curve]);
            cs->knots_offset[i_curve + 1] = cs->knots_offset[i_curve] + n_knots;
        }
    }

    return cs;
}

#endif // !__DATASET_LOADER_H__
