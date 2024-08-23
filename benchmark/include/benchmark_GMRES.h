#ifndef BENCHMARK_GMRES_H
#define BENCHMARK_GMRES_H

#include "benchmark.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "solvers/GMRES/GMRESSolve.h"

template <template <typename> typename TMatrix, typename TPrecision>
class NoProgressGMRESSolve: public cascade::GMRESSolve<TMatrix, TPrecision>
{
public:

    using cascade::GMRESSolve<TMatrix, TPrecision>::GMRESSolve;

    // Remove any progress towards solution
    void update_x_minimizing_res() override {
        GMRESSolve<TMatrix, TPrecision>::update_x_minimizing_res();
        this->typed_soln = this->init_guess_typed;
    }

};

class Benchmark_GMRES: public BenchmarkBase
{
public:

    std::function<NoFillMatrixSparse<double> (int, int)> make_norm_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {

        if (m != n) {
            throw std::runtime_error(
                "make_norm_A: require square dimensions"
            );
        }

        NoFillMatrixSparse<double> mat = NoFillMatrixSparse<double>::Random(
            BenchmarkBase::bundle, m, m,
            (sqrt(static_cast<double>(m))/static_cast<double>(m) > 1) ?
            1 :
            sqrt(static_cast<double>(m))/static_cast<double>(m)
        );
        mat.normalize_magnitude();

        return mat;

    };

    template <typename TPrecision>
    void execute_GMRES(
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        TypedLinearSystem<NoFillMatrixSparse, TPrecision> typed_lin_sys(
            &gen_lin_sys
        );
        SolveArgPkg args(
            gmres_iters, SolveArgPkg::default_max_inner_iter, 0
        );

        clock.clock_start();
        NoProgressGMRESSolve<NoFillMatrixSparse, TPrecision> gmres(
            &typed_lin_sys, 0., args
        );
        gmres.solve();
        clock.clock_stop();

        ASSERT_EQ(gmres.get_iteration(), gmres_iters);

    }

    template <typename TPrecision>
    MatrixDense<TPrecision> make_extrap_A(int m, int n) {

        if (m != n) {
            throw std::runtime_error(
                "make_extrap_A: require square dimensions"
            );
        }

        return MatrixDense<TPrecision>::Random(
            BenchmarkBase::bundle, m, dense_subset_cols+2
        );

    }

    template <typename TPrecision>
    void gmres_extrap_data_exec_func(
        Benchmark_AccumClock &clock, MatrixDense<TPrecision> &A
    ) {

        Vector<TPrecision> vec_m = Vector<TPrecision>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        Vector<TPrecision> vec_n = Vector<TPrecision>::Random(
            BenchmarkBase::bundle, A.cols()
        );
        MatrixDense<TPrecision> square_small_UT_A(
            MatrixDense<TPrecision>::Random_UT(
                BenchmarkBase::bundle, gmres_iters, gmres_iters
            )
        );
        Vector<TPrecision> small_b = Vector<TPrecision>::Random(
            BenchmarkBase::bundle, gmres_iters
        );
        Scalar<TPrecision> scalar(static_cast<TPrecision>(2.));

        clock.clock_start();

        for (int k=0; k<gmres_iters; k++) {

            // Dominant orthogonalization calculation operations
            A.transpose_prod_subset_cols(0, gmres_iters, vec_m);
            vec_m - A.mult_subset_cols(0, gmres_iters, small_b);
            A.transpose_prod_subset_cols(0, gmres_iters, vec_m);
            vec_m - A.mult_subset_cols(0, gmres_iters, small_b);

            // Dominant QR factorization update calculation operations
            square_small_UT_A.transpose_prod(small_b);
            small_b*scalar + small_b*scalar;
            small_b*scalar - small_b*scalar;

            // Dominant update x calculation operations
            square_small_UT_A.back_sub(small_b);

        }

        clock.clock_stop();

    };

};

#endif