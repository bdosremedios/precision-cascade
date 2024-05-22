#include "include/benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Benchmark_MP_GMRES_IR_Sparse: public Benchmark_Nested_GMRES {};

TEST_F(Benchmark_MP_GMRES_IR_Sparse, MP_GMRES_IR_BENCHMARK) {
    
    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(TestBase::bundle, A.rows());

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        SolveArgPkg args(nested_outer_iter, nested_inner_iter, 0.);

        clock.clock_start();
        SimpleConstantThreshold<NoFillMatrixSparse> mp_restarted_gmres(&gen_lin_sys, args);
        mp_restarted_gmres.solve();
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr, make_norm_A, execute_func, "mp_gmres_ir"
    );

}