#ifndef BENCHMARK_FP_GMRES_IR_H
#define BENCHMARK_FP_GMRES_IR_H

#include "benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class Benchmark_FP_GMRES_IR: public Benchmark_Nested_GMRES
{
public:

    template <typename TPrecision>
    void execute_fp_gmres_ir(
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        TypedLinearSystem<NoFillMatrixSparse, TPrecision> typed_lin_sys(
            &gen_lin_sys
        );
        SolveArgPkg args(nested_gmres_outer_iters, nested_gmres_inner_iters, 0.);

        clock.clock_start();
        FP_GMRES_IR_Solve<NoFillMatrixSparse, TPrecision> fp_restarted_gmres(
            &typed_lin_sys, 0., args
        );
        fp_restarted_gmres.solve();
        clock.clock_stop();

    };

};

#endif