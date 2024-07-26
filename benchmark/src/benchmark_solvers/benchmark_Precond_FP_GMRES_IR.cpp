#include "benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "tools/arg_pkgs/PrecondArgPkg.h"
#include "preconditioners/ILUPreconditioner.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

#include "tools/read_matrix.h"

class Benchmark_Precond_FP_GMRES_IR: public Benchmark_Nested_GMRES {};

TEST_F(Benchmark_Precond_FP_GMRES_IR, ILU0_FP_GMRES_IR_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &z
    ) {

        // NoFillMatrixSparse<double> A(
        //     NoFillMatrixSparse<double>::Random(
        //         BenchmarkBase::bundle,
        //         1024, 1024,
        //         sqrt(static_cast<double>(1024))/static_cast<double>(1024)
        //     )
        // );
        NoFillMatrixSparse<double> A(cascade::read_matrixCSV<NoFillMatrixSparse, double>(
            BenchmarkBase::bundle,
            BenchmarkBase::data_dir / fs::path("..") / fs::path("..") /
            fs::path("test") / fs::path("data") / fs::path("solve_matrices") /
            fs::path("ilu_sparse_A.csv")
        ));
        NoFillMatrixSparse<double> L(cascade::read_matrixCSV<NoFillMatrixSparse, double>(
            BenchmarkBase::bundle,
            BenchmarkBase::data_dir / fs::path("..") / fs::path("..") /
            fs::path("test") / fs::path("data") / fs::path("solve_matrices") /
            fs::path("ilu_sparse_L.csv")
        ));
        NoFillMatrixSparse<double> U(cascade::read_matrixCSV<NoFillMatrixSparse, double>(
            BenchmarkBase::bundle,
            BenchmarkBase::data_dir / fs::path("..") / fs::path("..") /
            fs::path("test") / fs::path("data") / fs::path("solve_matrices") /
            fs::path("ilu_sparse_U.csv")
        ));

        Vector<double> x_soln = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        TypedLinearSystem<NoFillMatrixSparse, double> typed_lin_sys(&gen_lin_sys);
        // SolveArgPkg args(nested_gmres_outer_iters, nested_gmres_inner_iters, 0.);
        SolveArgPkg args(nested_gmres_inner_iters, -1, 0.);

        clock.clock_start();
        std::shared_ptr<ILUPreconditioner<NoFillMatrixSparse, double>> matlab_ilu(
            new ILUPreconditioner<NoFillMatrixSparse, double>(L, U)
        );
        std::shared_ptr<ILUPreconditioner<NoFillMatrixSparse, double>> ilu(
            new ILUPreconditioner<NoFillMatrixSparse, double>(A)
        );
        Vector<double> vec(Vector<double>::Ones(BenchmarkBase::bundle, A.rows()));
        PrecondArgPkg<NoFillMatrixSparse, double> precond_args(ilu);
        // std::cout << A.get_info_string() << std::endl;
        // std::cout << ((MatrixDense<double>(ilu->get_L()) *
        //                MatrixDense<double>(ilu->get_U())) -
        //               MatrixDense<double>(A)).norm().get_scalar()
        //           << std::endl;
        // FP_GMRES_IR_Solve<NoFillMatrixSparse, double> fp_restarted_gmres(
        //     &typed_lin_sys, 0., args, precond_args
        // );
        GMRESSolve<NoFillMatrixSparse, double> fp_restarted_gmres(
            &typed_lin_sys, 0., args//, precond_args
        );
        fp_restarted_gmres.solve();
        fp_restarted_gmres.view_relres_plot();
        Vector<double> vec_1(MatrixDense<double>(matlab_ilu->get_L()).frwd_sub(vec));
        Vector<double> vec_2(MatrixDense<double>(ilu->get_L()).frwd_sub(vec));
        Vector<double> vec_3(MatrixDense<double>(matlab_ilu->get_U()).back_sub(vec_1));
        Vector<double> vec_4(MatrixDense<double>(ilu->get_U()).back_sub(vec_2));
        for (int k=0; k<A.rows(); ++k) {
            std::cout << (vec_3.get_elem(k) - vec_4.get_elem(k)).get_scalar()
                      << " ";
        }
        std::cout << std::endl << std::endl;
        // Vector<double> vec_1(MatrixDense<double>(matlab_ilu->get_L()).frwd_sub(vec));
        // Vector<double> vec_2(MatrixDense<double>(ilu->get_L()).frwd_sub(vec));
        // for (int k=0; k<A.rows(); ++k) {
        //     std::cout << (vec_1.get_elem(k) - vec_2.get_elem(k)).get_scalar()
        //               << " ";
        // }
        // std::cout << std::endl << std::endl;
        // Vector<double> vec_3(MatrixDense<double>(matlab_ilu->get_U()).back_sub(vec));
        // Vector<double> vec_4(MatrixDense<double>(ilu->get_U()).back_sub(vec));
        // for (int k=0; k<A.rows(); ++k) {
        //     std::cout << (vec_3.get_elem(k) - vec_4.get_elem(k)).get_scalar()
        //               << " ";
        // }
        // std::cout << std::endl << std::endl;
        clock.clock_stop();

    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        ilu_dims, make_norm_A, execute_func, "ilu0_fp_gmres_ir"
    );

}