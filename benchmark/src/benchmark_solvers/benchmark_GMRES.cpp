#include "benchmark_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/GMRES/GMRESSolve.h"

TEST_F(Benchmark_GMRES, GMRESSolve_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(BenchmarkBase::bundle, A.rows());

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

TEST_F(Benchmark_GMRES, GetExtrapolationData) {

    std::function<MatrixDense<double> (int, int)> make_A_m_n = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random(BenchmarkBase::bundle, m, dense_subset_cols+2);
    };

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {

        Vector<double> vec_m = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        Vector<double> vec_n = Vector<double>::Random(BenchmarkBase::bundle, A.cols());
        MatrixDense<double> square_small_A(
            MatrixDense<double>::Random_UT(BenchmarkBase::bundle, dense_subset_cols, dense_subset_cols)
        );
        Vector<double> small_b = Vector<double>::Random(BenchmarkBase::bundle, dense_subset_cols);

        clock.clock_start();
        A.transpose_prod_subset_cols(0, dense_subset_cols, vec_m);
        A.mult_subset_cols(0, dense_subset_cols, small_b);
        A.transpose_prod_subset_cols(0, dense_subset_cols, vec_m);
        A.mult_subset_cols(0, dense_subset_cols, small_b);
        square_small_A.back_sub(small_b);
        clock.clock_stop();

    };

    benchmark_exec_func<MatrixDense, double>(
        sparse_start, sparse_stop, sparse_incr, make_A_m_n, execute_func, "gmressolveextrapdata"
    );


}