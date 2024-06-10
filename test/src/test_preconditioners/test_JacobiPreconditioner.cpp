#include "test.h"

#include "preconditioners/JacobiPreconditioner.h"

class JacobiPreconditioner_Test: public TestBase
{
public:

    template< template <typename> typename TMatrix>
    void TestJacobiPreconditioner() {
        
        constexpr int n(45);
        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("A_inv_45.csv")
        ));
        JacobiPreconditioner<TMatrix, double> jacobi_precond(A);

        // Check compatibility with only 45
        ASSERT_TRUE(jacobi_precond.check_compatibility_left(n));
        ASSERT_TRUE(jacobi_precond.check_compatibility_right(n));
        ASSERT_FALSE(jacobi_precond.check_compatibility_left(6));
        ASSERT_FALSE(jacobi_precond.check_compatibility_right(6));
        ASSERT_FALSE(jacobi_precond.check_compatibility_left(100));
        ASSERT_FALSE(jacobi_precond.check_compatibility_right(100));

        Vector<double> orig_test_vec(Vector<double>::Random(
            TestBase::bundle, n
        ));
        Vector<double> test_vec(jacobi_precond.action_inv_M(orig_test_vec));
        Vector<double> target_vec(orig_test_vec);
        for (int i=0; i<orig_test_vec.rows(); ++i) {
            target_vec.set_elem(i, target_vec.get_elem(i)/A.get_elem(i, i));
        }

        ASSERT_VECTOR_NEAR(
            test_vec,
            target_vec,
            4*A.get_max_mag_elem().get_scalar()*Tol<double>::roundoff_T()
        );

    }

};

TEST_F(JacobiPreconditioner_Test, TestJacobiPreconditioner_PRECONDITIONER) {
    TestJacobiPreconditioner<MatrixDense>();
    TestJacobiPreconditioner<NoFillMatrixSparse>();
}