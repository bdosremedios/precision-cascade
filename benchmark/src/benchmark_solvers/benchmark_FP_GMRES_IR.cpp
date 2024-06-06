#include "benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class Benchmark_FP_GMRES_IR: public Benchmark_Nested_GMRES {};

TEST_F(Benchmark_FP_GMRES_IR, FP_GMRES_IR_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(BenchmarkBase::bundle, A.rows());

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        TypedLinearSystem<NoFillMatrixSparse, double> typed_lin_sys(&gen_lin_sys);
        SolveArgPkg args(nested_outer_iter, nested_inner_iter, 0.);

        clock.clock_start();
        FP_GMRES_IR_Solve fp_restarted_gmres(&typed_lin_sys, 0., args);
        fp_restarted_gmres.solve();
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse>(
        sparse_start, sparse_stop, sparse_incr, make_norm_A, execute_func, "fp_gmres_ir"
    );

}