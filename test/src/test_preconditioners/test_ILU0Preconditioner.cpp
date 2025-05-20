#include "test.h"

#include "preconditioners/implemented_preconditioners.h"

class ILU0_Test: public TestBase
{
public:

    template <template <typename> typename TMatrix>
    void TestMatchesDenseLU() {

        // Test that using a completely dense matrix one just gets LU
        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv")
        ));
        ILUPreconditioner<TMatrix, double> ilu_precond(A);
        TMatrix<double> test(
            MatrixDense<double>(ilu_precond.get_L()) *
            MatrixDense<double>(ilu_precond.get_U()) -
            MatrixDense<double>(A)
        );

        ASSERT_MATRIX_ZERO(test, Tol<double>::dbl_ilu_elem_tol());

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        TMatrix<double> L(
            read_matrixCSV<TMatrix, double>(
                TestBase::bundle, solve_matrix_dir / fs::path("ilu_L.csv")
            )
        );
        TMatrix<double> U(
            read_matrixCSV<TMatrix, double>(
                TestBase::bundle, solve_matrix_dir / fs::path("ilu_U.csv")
            )
        );

        MatrixDense<double> delta_A(
            (MatrixDense<double>(ilu_precond.get_L()) *
             MatrixDense<double>(ilu_precond.get_U())) -
             MatrixDense<double>(A)
        );

        double norm_diff = delta_A.norm().get_scalar();
        ASSERT_NEAR(
            (MatrixDense<double>(ilu_precond.get_L()) *
             MatrixDense<double>(ilu_precond.get_U()) -
             MatrixDense<double>(A)).norm().get_scalar(),
            norm_diff,
            1e-10
        );

        // Test compliant with Higham 2002 ch.9 Thm 9.3
        MatrixDense<double> dense_gen_L(ilu_precond.get_L());
        MatrixDense<double> dense_gen_U(ilu_precond.get_U());
        dense_gen_L.abs();
        dense_gen_U.abs();
        MatrixDense<double> dense_comp(dense_gen_L*dense_gen_U);
        dense_comp *= Scalar<double>(Tol<double>::gamma(dense_comp.rows()));
        delta_A.abs();
        ASSERT_MATRIX_LT(delta_A, dense_comp);

        ASSERT_MATRIX_NEAR(
            ilu_precond.get_L(),
            L,
            mat_max_mag(L)*Tol<double>::dbl_ilu_elem_tol()
        );
        ASSERT_MATRIX_NEAR(
            ilu_precond.get_U(),
            U,
            mat_max_mag(U)*Tol<double>::dbl_ilu_elem_tol()
        );

    }

    template <template <typename> typename TMatrix>
    void TestMatchesSparseILU0() {

        // Test sparsity matches zero pattern for ILU0 on sparse A
        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_A.csv")
        ));
        ILUPreconditioner<TMatrix, double> ilu_precond(A);

        ASSERT_MATRIX_SAMESPARSITY(
            ilu_precond.get_L(), A, Tol<double>::roundoff()
        );
        ASSERT_MATRIX_SAMESPARSITY(
            ilu_precond.get_U(), A, Tol<double>::roundoff()
        );

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        TMatrix<double> L(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_L.csv")
        ));
        TMatrix<double> U(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_U.csv")
        ));

        double norm_diff = (
            (MatrixDense<double>(L) * MatrixDense<double>(U) -
             MatrixDense<double>(A)).norm().get_scalar()
        );
        ASSERT_NEAR(
            (MatrixDense<double>(ilu_precond.get_L()) *
             MatrixDense<double>(ilu_precond.get_U()) -
             MatrixDense<double>(A)).norm().get_scalar(),
            norm_diff,
            1e-10
        );
        ASSERT_MATRIX_NEAR(
            ilu_precond.get_L(), L, Tol<double>::dbl_ilu_elem_tol()
        );
        ASSERT_MATRIX_NEAR(
            ilu_precond.get_U(), U, Tol<double>::dbl_ilu_elem_tol()
        );

    }

};

TEST_F(ILU0_Test, TestMatchesDenseLU_PRECONDITIONER) {
    TestMatchesDenseLU<MatrixDense>();
    TestMatchesDenseLU<NoFillMatrixSparse>();
}

TEST_F(ILU0_Test, TestMatchesSparseILU0_PRECONDITIONER) {
    TestMatchesSparseILU0<MatrixDense>();
    TestMatchesSparseILU0<NoFillMatrixSparse>();
}