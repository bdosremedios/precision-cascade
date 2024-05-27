#include "include/benchmark_Sparse.h"

#include "preconditioners/ILUPreconditioner.h"

class Benchmark_ILUPreconditioner: public Benchmark_Sparse {};

TEST_F(Benchmark_ILUPreconditioner, Double_ILUPreconditioner_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(TestBase::bundle, A.rows());

        clock.clock_start();
        ILUPreconditioner<NoFillMatrixSparse, double> ilu(A, true);
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse>(
        ilu_start, ilu_stop, ilu_incr, make_A_dbl, execute_func, "ilu0_precond_dbl"
    );

}