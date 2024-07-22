#include "benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "tools/arg_pkgs/PrecondArgPkg.h"
#include "preconditioners/ILUPreconditioner.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class Benchmark_Precond_FP_GMRES_IR: public Benchmark_Nested_GMRES {};

TEST_F(Benchmark_Precond_FP_GMRES_IR, ILU0_FP_GMRES_IR_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        TypedLinearSystem<NoFillMatrixSparse, double> typed_lin_sys(&gen_lin_sys);
        SolveArgPkg args(nested_gmres_outer_iters, nested_gmres_inner_iters, 0.);

        clock.clock_start();
        PrecondArgPkg<NoFillMatrixSparse, double> precond_args(
            std::make_shared<ILUPreconditioner<NoFillMatrixSparse, double>>(A)
        );
        FP_GMRES_IR_Solve fp_restarted_gmres(
            &typed_lin_sys, 0., args, precond_args
        );
        fp_restarted_gmres.solve();
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse>(
        ilu_dims, make_norm_A, execute_func, "ilu0_fp_gmres_ir"
    );

}