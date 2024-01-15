#include "../test.h"

class Substitution_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestBackwardSubstitution() {

        constexpr int n(90);
        M<double> U_tri(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("U_tri_90.csv"))
        );
        MatrixVector<double> x_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("x_tri_90.csv"))
        );
        MatrixVector<double> Ub_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("Ub_tri_90.csv"))
        );
    
        MatrixVector<double> test_soln(U_tri.back_sub(Ub_tri));

        ASSERT_VECTOR_NEAR(test_soln, x_tri, Tol<double>::dbl_substitution_tol());

    }

    template <template <typename> typename M>
    void TestForwardSubstitution() {

        constexpr int n(90);
        M<double> L_tri(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("L_tri_90.csv"))
        );
        MatrixVector<double> x_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("x_tri_90.csv"))
        );
        MatrixVector<double> Lb_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("Lb_tri_90.csv"))
        );
    
        MatrixVector<double> test_soln(L_tri.frwd_sub(Lb_tri));

        ASSERT_VECTOR_NEAR(test_soln, x_tri, Tol<double>::dbl_substitution_tol());

    }

};

TEST_F(Substitution_Test, TestBackwardSubstitution) {
    TestBackwardSubstitution<MatrixDense>();
    // TestBackwardSubstitution<MatrixSparse>();
}

TEST_F(Substitution_Test, TestForwardSubstitution) {
    TestForwardSubstitution<MatrixDense>();
//     // TestForwardSubstitution<MatrixSparse>();
}