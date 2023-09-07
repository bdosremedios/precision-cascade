#include "test.h"

#include "tools/Substitution.h"

class SubstitutionTest: public TestBase {};

TEST_F(SubstitutionTest, TestBackwardSubstitution) {

    constexpr int n(90);
    Matrix<double, Dynamic, Dynamic> U_tri(read_matrix_csv<double>(solve_matrix_dir + "U_tri_90.csv"));
    Matrix<double, Dynamic, 1> x_tri(read_matrix_csv<double>(solve_matrix_dir + "x_tri_90.csv"));
    Matrix<double, Dynamic, 1> Ub_tri(read_matrix_csv<double>(solve_matrix_dir + "Ub_tri_90.csv"));
    Matrix<double, n, 1> test_soln(back_substitution(U_tri, Ub_tri));

    for (int i=0; i<n; ++i) {
        ASSERT_NEAR(test_soln[i], x_tri[i], u_dbl);
    }
    ASSERT_NEAR((Ub_tri-U_tri*test_soln).norm(), 0, u_dbl);

}

TEST_F(SubstitutionTest, TestForwardSubstitution) {

    constexpr int n(90);
    Matrix<double, Dynamic, Dynamic> L_tri(read_matrix_csv<double>(solve_matrix_dir + "L_tri_90.csv"));
    Matrix<double, Dynamic, 1> x_tri(read_matrix_csv<double>(solve_matrix_dir + "x_tri_90.csv"));
    Matrix<double, Dynamic, 1> Lb_tri(read_matrix_csv<double>(solve_matrix_dir + "Lb_tri_90.csv"));
    Matrix<double, n, 1> test_soln(frwd_substitution(L_tri, Lb_tri));

    for (int i=0; i<n; ++i) {
        ASSERT_NEAR(test_soln[i], x_tri[i], u_dbl);
    }
    ASSERT_NEAR((Lb_tri-L_tri*test_soln).norm(), 0, u_dbl);

}