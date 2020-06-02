#ifndef __BSPLINE_CUH__
#define __BSPLINE_CUH__

#include <gcurval/base.h>

template <int degree = 3, typename Dt = float>
GCURVAL_HD_FUNC int find_span(Dt u, const Dt* knot_vector, int m)
{
    if (u == Dt(1.0))
    {
        return m - degree - 1; // n
    }

    int i = degree;
    for (; i < m - degree; i++)
    {
        if ((u >= knot_vector[i]) && (u < knot_vector[i + 1]))
        {
            break;
        }
    }

    return i;
}

template <int degree = 3, typename Dt = float>
GCURVAL_HD_FUNC void b_spline_func_value(Dt u, int span, const Dt* knot_vector, Dt* func_values, Dt* left, Dt* right)
{
    Dt saved, temp;

    func_values[0] = Dt(1.0);

#pragma unroll
    for (int j = 1; j <= degree; j++)
    {
        left[j] = u - knot_vector[span + 1 - j];
        right[j] = knot_vector[span + j] - u;
        saved = 0.0;
        for (int r = 0; r < j; r++)
        {
            temp = func_values[r] / (right[r + 1] + left[j - r]);
            func_values[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        func_values[j] = saved;
    }
}

// derivative order 1
template <int degree = 3, typename Dt = float>
GCURVAL_HD_FUNC void b_spline_derv_o1(Dt u, int span, const Dt* knot_vector, Dt* func_values, Dt* left, Dt* right)
{
    Dt saved, temp;

    func_values[0] = Dt(1.0);

    // function values
    for (int j = 1; j < degree; j++)
    {
        left[j] = u - knot_vector[span + 1 - j];
        right[j] = knot_vector[span + j] - u;
        saved = 0.0;
        for (int r = 0; r < j; r++)
        {
            temp = func_values[r] / (right[r + 1] + left[j - r]);
            func_values[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        func_values[j] = saved;
    }

    // derivative values
    left[degree] = u - knot_vector[span + 1 - degree];
    right[degree] = knot_vector[span + degree] - u;
    saved = 0.0;
    for (int r = 0; r < degree; r++)
    {
        temp = degree * func_values[r] / (right[r + 1] + left[degree - r]);
        func_values[r] = saved - temp; // Note the '-'

        saved = temp;
    }
    func_values[degree] = saved;

}
// curve dot
template <typename Dt = float, typename PointType, int degree = 3>
GCURVAL_HD_FUNC PointType to_point(int span, const PointType* ctrl_ptrs, const Dt* fun_value)
{
    PointType result = ctrl_ptrs[span - degree] * fun_value[0];
    for (int i = 1; i <= degree; i++)
    {
        result += ctrl_ptrs[span - degree + i] * fun_value[i];
    }
    return result;
}


#endif // !__BSPLINE_CUH__