#include "include/benchmark_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/GMRES/GMRESSolve.h"

TEST_F(Benchmark_GMRES, GMRESSolve_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(TestBase::bundle, A.rows());

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        TypedLinearSystem<NoFillMatrixSparse, double> typed_lin_sys(&gen_lin_sys);
        SolveArgPkg args(gmressolve_iters, SolveArgPkg::default_max_inner_iter, 0);

        clock.clock_start();
        GMRESSolve gmres(&typed_lin_sys, 0., args);
        gmres.solve();
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr, make_norm_A, execute_func, "gmressolve"
    );

}