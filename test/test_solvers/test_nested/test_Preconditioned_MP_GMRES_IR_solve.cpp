#include "../../test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Preconditioned_MP_GMRES_IR_Test: public TestBase
{
public:

    const SolveArgPkg solve_args = SolveArgPkg(80, 10, Tol<double>::krylov_conv_tol());

    template <template <typename> typename M>
    void PreconditionedSolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args,
        const double &conv_tol
    ) {

        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, A_file_path));
        Vector<double> b(read_matrixCSV<Vector, double>(TestBase::bundle, b_file_path));

        GenericLinearSystem<M> gen_lin_sys(A, b);

        RestartCount<M> gmres_ir(&gen_lin_sys, args);
        gmres_ir.solve();

        PrecondArgPkg<M, double> precond_args(
            std::make_shared<ILUPreconditioner<M, double>>(A, true)
        );
        RestartCount<M> precond_gmres_ir(&gen_lin_sys, args, precond_args);
        precond_gmres_ir.solve();

        if (*show_plots) { precond_gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(precond_gmres_ir.check_converged());
        EXPECT_LE(precond_gmres_ir.get_relres(), Tol<double>::krylov_conv_tol());
        EXPECT_LT(precond_gmres_ir.get_iteration(), gmres_ir.get_iteration());

    }

};

TEST_F(Preconditioned_MP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    
    PreconditionedSolveTest<MatrixDense>(
        A_path, b_path, solve_args, Tol<double>::nested_krylov_conv_tol()
    );
    PreconditionedSolveTest<NoFillMatrixSparse>(
        A_path, b_path, solve_args, Tol<double>::nested_krylov_conv_tol()
    );

}

TEST_F(Preconditioned_MP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense>(
        A_path, b_path, solve_args, Tol<double>::nested_krylov_conv_tol()
    );
    PreconditionedSolveTest<NoFillMatrixSparse>(
        A_path, b_path, solve_args, Tol<double>::nested_krylov_conv_tol()
    );

}
