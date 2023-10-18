#include "../test.h"

#include "tools/Substitution.h"

class SubstitutionTest: public TestBase
{
public:

    template <template <typename> typename M>
    void TestBackwardSubstitution() {

        constexpr int n(90);
        M<double> U_tri = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("U_tri_90.csv"));
        MatrixVector<double> x_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_tri_90.csv"));
        MatrixVector<double> Ub_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("Ub_tri_90.csv"));
        MatrixVector<double> test_soln = back_substitution(U_tri, Ub_tri);

        for (int i=0; i<n; ++i) { ASSERT_NEAR(test_soln[i], x_tri[i], u_dbl); }
        ASSERT_NEAR((Ub_tri-U_tri*test_soln).norm(), 0, u_dbl);

    }

    template <template <typename> typename M>
    void TestForwardSubstitution() {

        constexpr int n(90);
        M<double> L_tri = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("L_tri_90.csv"));
        MatrixVector<double> x_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_tri_90.csv"));
        MatrixVector<double> Lb_tri = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("Lb_tri_90.csv"));
        MatrixVector<double> test_soln = frwd_substitution(L_tri, Lb_tri);

        for (int i=0; i<n; ++i) { ASSERT_NEAR(test_soln[i], x_tri[i], u_dbl); }
        ASSERT_NEAR((Lb_tri-L_tri*test_soln).norm(), 0, u_dbl);

    }

};

TEST_F(SubstitutionTest, TestBackwardSubstitution_Dense) { TestBackwardSubstitution<MatrixDense>(); }
TEST_F(SubstitutionTest, TestBackwardSubstitution_Sparse) { TestBackwardSubstitution<MatrixSparse>(); }

TEST_F(SubstitutionTest, TestForwardSubstitution_Dense) { TestForwardSubstitution<MatrixDense>(); }
TEST_F(SubstitutionTest, TestForwardSubstitution_Sparse) { TestForwardSubstitution<MatrixSparse>(); }