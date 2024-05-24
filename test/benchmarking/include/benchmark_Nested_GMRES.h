#ifndef BENCHMARK_NESTED_GMRES_H
#define BENCHMARK_NESTED_GMRES_H

#include "benchmark_GMRES.h"

class Benchmark_Nested_GMRES: public Benchmark_GMRES
{
public:

    const int nested_outer_iter = 10;
    const int nested_inner_iter = gmressolve_iters;

};

#endif