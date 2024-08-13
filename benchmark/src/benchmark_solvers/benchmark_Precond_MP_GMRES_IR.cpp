#include "benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "tools/arg_pkgs/PrecondArgPkg.h"
#include "preconditioners/ILUPreconditioner.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Benchmark_Precond_MP_GMRES_IR: public Benchmark_Nested_GMRES {};

TEST_F(Benchmark_Precond_MP_GMRES_IR, ILU0_MP_GMRES_IR_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        SolveArgPkg args(
            nested_gmres_outer_iters, nested_gmres_inner_iters, 0.
        );

        clock.clock_start();
        PrecondArgPkg<NoFillMatrixSparse, double> precond_args(
            std::make_shared<ILUPreconditioner<NoFillMatrixSparse, double>>(A)
        );
        OuterRestartCount<NoFillMatrixSparse> mp_restarted_gmres(
            &gen_lin_sys, args, precond_args
        );
        mp_restarted_gmres.solve();
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse>(
        ilu_dims, make_norm_A, execute_func, "ilu0_outer_restart_count"
    );

}