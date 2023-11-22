#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILUTP_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestEquivalentILUTNoDropAndDenseILU0() {

        // Test ILU(0) and ILUT(0) [No Dropping] Give the same dense decomp
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu0(A, u_dbl, false);
        ILU<M, double> ilut0(A, 0., n, u_dbl, false);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                if (A.coeff(i, j) == 0.) {
                    ASSERT_EQ(ilu0.get_L().coeff(i, j), ilut0.get_L().coeff(i, j));
                    ASSERT_EQ(ilu0.get_U().coeff(i, j), ilut0.get_U().coeff(i, j));
                }
            }
        }

    }

    template<template <typename> typename M>
    void TestILUTDropping() {

        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));

        // Check multiple rising thresholds ensuring that each ilu is closer to the matrix and that
        // all have correct form for L and U
        ILU<M, double> ilut1e_6(A, 1e-6, n, u_dbl, false);
        ILU<M, double> ilut1e_3(A, 1e-3, n, u_dbl, false);
        ILU<M, double> ilut1e_1(A, 1e-1, n, u_dbl, false);
        ILU<M, double> ilut1e_0(A, 1., n, u_dbl, false);

        // Test correct L and U triangularity
        for (int i=0; i<n; ++i) {
            for (int j=i+1; j<n; ++j) {
                ASSERT_NEAR(ilut1e_6.get_L().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut1e_3.get_L().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut1e_1.get_L().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut1e_0.get_L().coeff(i, j), 0, dbl_error_acc);
            }
        }
        for (int i=0; i<n; ++i) {
            for (int j=0; j<i; ++j) {
                ASSERT_NEAR(ilut1e_6.get_U().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut1e_3.get_U().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut1e_1.get_U().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut1e_0.get_U().coeff(i, j), 0, dbl_error_acc);
            }
        }

        // Test that each lower threshold is better than the higher one
        EXPECT_LE((A-ilut1e_6.get_L()*ilut1e_6.get_U()).norm(),
                  (A-ilut1e_3.get_L()*ilut1e_3.get_U()).norm());
        EXPECT_LE((A-ilut1e_3.get_L()*ilut1e_3.get_U()).norm(),
                  (A-ilut1e_1.get_L()*ilut1e_1.get_U()).norm());
        EXPECT_LE((A-ilut1e_1.get_L()*ilut1e_1.get_U()).norm(),
                  (A-ilut1e_0.get_L()*ilut1e_0.get_U()).norm());

        // Test that each higher threshold has more zeros
        EXPECT_LE(count_zeros(ilut1e_6.get_L(), u_dbl),
                  count_zeros(ilut1e_3.get_L(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_3.get_L(), u_dbl),
                  count_zeros(ilut1e_1.get_L(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_1.get_L(), u_dbl),
                  count_zeros(ilut1e_0.get_L(), u_dbl));

    }

    template<template <typename> typename M>
    void TestILUTDroppingLimits() {

        // Test that max dropping just gives the diagonal since everything else gets dropped
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));
        ILU<M, double> ilu_all_drop(A, DBL_MAX, n, u_dbl, false);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                if (i != j) {
                    ASSERT_NEAR(ilu_all_drop.get_L().coeff(i, j), 0., dbl_error_acc);
                    ASSERT_NEAR(ilu_all_drop.get_U().coeff(i, j), 0., dbl_error_acc);
                } else {
                    ASSERT_NEAR(ilu_all_drop.get_L().coeff(i, j), 1., dbl_error_acc);
                    ASSERT_NEAR(ilu_all_drop.get_U().coeff(i, j), A.coeff(i, j), dbl_error_acc);
                }
            }
        }

    }

    // template<template <typename> typename M>
    // void TestILUTPMatchMATLAB1e_6() {

    //     constexpr int n(8);
    //     M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));
    //     M<double> target_L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilutp_L_1e_6.csv"));
    //     M<double> target_U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilutp_U_1e_6.csv"));
    //     M<double> target_P = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilutp_P_1e_6.csv"));

    //     ILU<M, double> ilut1e_6(A, 1e-6, n, u_dbl, true);

    //     ASSERT_MATRIX_NEAR<M, double>(ilut1e_6.get_L(), target_L, u_dbl);
    //     ASSERT_MATRIX_NEAR<M, double>(ilut1e_6.get_U(), target_U, u_dbl);
    //     ASSERT_MATRIX_NEAR<M, double>(ilut1e_6.get_P(), target_P, u_dbl);

    //     cout << "A: " << endl;
    //     A.print();
    //     cout << "test_L: " << endl;
    //     ilut1e_6.get_L().print();
    //     cout << "target_L: " << endl;
    //     target_L.print();
    //     cout << "diff_L: " << endl;
    //     (target_L-ilut1e_6.get_L()).print();
    //     cout << "test_U: " << endl;
    //     ilut1e_6.get_U().print();
    //     cout << "target_U: " << endl;
    //     target_U.print();
    //     cout << "diff_U: " << endl;
    //     (target_U-ilut1e_6.get_U()).print();
    //     cout << "test_P: " << endl;
    //     ilut1e_6.get_P().print();
    //     cout << "target_P: " << endl;
    //     target_P.print();
    //     cout << "diff_P: " << endl;
    //     (target_P-ilut1e_6.get_P()).print();

    // }

};

TEST_F(ILUTP_Test, TestEquivalentILUTNoDropAndDenseILU0_Dense) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixDense>();
}
TEST_F(ILUTP_Test, TestEquivalentILUTNoDropAndDenseILU0_Sparse) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixSparse>();
}

TEST_F(ILUTP_Test, TestILUTDropping_Dense) { TestILUTDropping<MatrixDense>(); }
TEST_F(ILUTP_Test, TestILUTDropping_Sparse) { TestILUTDropping<MatrixSparse>(); }

TEST_F(ILUTP_Test, TestILUTDroppingLimits_Dense) { TestILUTDroppingLimits<MatrixDense>(); }
TEST_F(ILUTP_Test, TestILUTDroppingLimits_Sparse) { TestILUTDroppingLimits<MatrixSparse>(); }

// TEST_F(ILUTP_Test, TestILUTPMatchMATLAB1e_6_Dense) { TestILUTPMatchMATLAB1e_6<MatrixDense>(); }
// TEST_F(ILUTP_Test, TestILUTPMatchMATLAB1e_6_Sparse) { TestILUTPMatchMATLAB1e_6<MatrixSparse>(); }