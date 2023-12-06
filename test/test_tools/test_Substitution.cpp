#include "../test.h"

#include "tools/Substitution.h"

class Substitution_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestBackwardSubstitution() {

        constexpr int n(90);
        M<double> U_tri = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("U_tri_90.csv"));
        MatrixVector<double> x_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_tri_90.csv"));
        MatrixVector<double> Ub_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("Ub_tri_90.csv"));
    
        MatrixVector<double> test_soln = back_substitution(U_tri, Ub_tri);

        ASSERT_VECTOR_NEAR(test_soln, x_tri, Tol<double>::roundoff());
        ASSERT_VECTOR_NEAR(Ub_tri, U_tri*test_soln, Tol<double>::gamma(n));

    }

    template <template <typename> typename M>
    void TestForwardSubstitution() {

        constexpr int n(90);
        M<double> L_tri = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("L_tri_90.csv"));
        MatrixVector<double> x_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_tri_90.csv"));
        MatrixVector<double> Lb_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("Lb_tri_90.csv"));
    
        MatrixVector<double> test_soln = frwd_substitution(L_tri, Lb_tri);

        ASSERT_VECTOR_NEAR(test_soln, x_tri, Tol<double>::roundoff());
        ASSERT_VECTOR_NEAR(Lb_tri, L_tri*test_soln, Tol<double>::gamma(n));

    }

};

TEST_F(Substitution_Test, TestBackwardSubstitution_Dense) { TestBackwardSubstitution<MatrixDense>(); }
TEST_F(Substitution_Test, TestBackwardSubstitution_Sparse) { TestBackwardSubstitution<MatrixSparse>(); }

TEST_F(Substitution_Test, TestForwardSubstitution_Dense) { TestForwardSubstitution<MatrixDense>(); }
TEST_F(Substitution_Test, TestForwardSubstitution_Sparse) { TestForwardSubstitution<MatrixSparse>(); }