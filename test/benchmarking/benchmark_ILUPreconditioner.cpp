#include "include/benchmark_Sparse.h"

#include "preconditioners/ILUPreconditioner.h"

class Benchmark_ILUPreconditioner: public Benchmark_Sparse {};

TEST_F(Benchmark_ILUPreconditioner, Double_ILU0_Preconditioner_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        clock.clock_start();
        ILUPreconditioner<NoFillMatrixSparse, double> ilu(A);
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse>(
        ilu_start, ilu_stop, ilu_incr, make_A_dbl, execute_func, "ilu0_precond_dbl"
    );

}

TEST_F(Benchmark_ILUPreconditioner, Double_ILUT_10e_4_20_Preconditioner_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        clock.clock_start();
        ILUPreconditioner<NoFillMatrixSparse, double> ilu(A, 1e-4, 20, false);
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse>(
        ilu_start, ilu_stop, ilu_incr, make_A_dbl, execute_func, "ilutp_em4_20_precond_dbl"
    );

}

TEST_F(Benchmark_ILUPreconditioner, Double_ILUT_10e_6_20_Preconditioner_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        clock.clock_start();
        ILUPreconditioner<NoFillMatrixSparse, double> ilu(A, 1e-6, 20, false);
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse>(
        ilu_start, ilu_stop, ilu_incr, make_A_dbl, execute_func, "ilutp_em6_20_precond_dbl"
    );

}