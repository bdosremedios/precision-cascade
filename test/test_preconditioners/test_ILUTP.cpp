#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILUTP_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestEquivalentILUTNoDropAndDenseILU0() {

        // Test ILU(0) and ILUT(0, n) [No Dropping] Give the same dense decomp
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu0(A, u_dbl, false);
        ILU<M, double> ilut0(A, 0., n, u_dbl, false);

        ASSERT_MATRIX_EQ(ilu0.get_L(), ilut0.get_L());
        ASSERT_MATRIX_EQ(ilu0.get_U(), ilut0.get_U());

    }

    template<template <typename> typename M>
    void TestILUTDropping(bool pivot) {

        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));

        // Check multiple rising thresholds ensuring that each ilu is closer to the matrix and that
        // all have correct form for L and U
        ILU<M, double> ilut1e_4(A, 1e-4, n, u_dbl, pivot);
        ILU<M, double> ilut1e_3(A, 1e-3, n, u_dbl, pivot);
        ILU<M, double> ilut1e_2(A, 1e-2, n, u_dbl, pivot);
        ILU<M, double> ilut1e_1(A, 1e-1, n, u_dbl, pivot);
        ILU<M, double> ilut1e_0(A, 1e-0, n, u_dbl, pivot);

        ASSERT_MATRIX_LOWTRI(ilut1e_4.get_L(), dbl_error_acc);
        ASSERT_MATRIX_LOWTRI(ilut1e_3.get_L(), dbl_error_acc);
        ASSERT_MATRIX_LOWTRI(ilut1e_2.get_L(), dbl_error_acc);
        ASSERT_MATRIX_LOWTRI(ilut1e_1.get_L(), dbl_error_acc);
        ASSERT_MATRIX_LOWTRI(ilut1e_0.get_L(), dbl_error_acc);

        ASSERT_MATRIX_UPPTRI(ilut1e_4.get_U(), dbl_error_acc);
        ASSERT_MATRIX_UPPTRI(ilut1e_3.get_U(), dbl_error_acc);
        ASSERT_MATRIX_UPPTRI(ilut1e_2.get_U(), dbl_error_acc);
        ASSERT_MATRIX_UPPTRI(ilut1e_1.get_U(), dbl_error_acc);
        ASSERT_MATRIX_UPPTRI(ilut1e_0.get_U(), dbl_error_acc);

        // Test that each lower threshold is better than the higher one w.r.t
        // Frobenius norm
        EXPECT_LE((A-ilut1e_4.get_L()*ilut1e_4.get_U()).norm(),
                  (A-ilut1e_3.get_L()*ilut1e_3.get_U()).norm());
        EXPECT_LE((A-ilut1e_3.get_L()*ilut1e_3.get_U()).norm(),
                  (A-ilut1e_2.get_L()*ilut1e_2.get_U()).norm());
        EXPECT_LE((A-ilut1e_2.get_L()*ilut1e_2.get_U()).norm(),
                  (A-ilut1e_1.get_L()*ilut1e_1.get_U()).norm());
        EXPECT_LE((A-ilut1e_1.get_L()*ilut1e_1.get_U()).norm(),
                  (A-ilut1e_0.get_L()*ilut1e_0.get_U()).norm());

        // Test that each higher threshold has more zeros
        EXPECT_LE(count_zeros(ilut1e_4.get_U(), u_dbl),
                  count_zeros(ilut1e_3.get_U(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_3.get_U(), u_dbl),
                  count_zeros(ilut1e_2.get_U(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_2.get_U(), u_dbl),
                  count_zeros(ilut1e_1.get_U(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_1.get_U(), u_dbl),
                  count_zeros(ilut1e_0.get_U(), u_dbl));

        EXPECT_LE(count_zeros(ilut1e_4.get_L(), u_dbl),
                  count_zeros(ilut1e_3.get_L(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_3.get_L(), u_dbl),
                  count_zeros(ilut1e_2.get_L(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_2.get_L(), u_dbl),
                  count_zeros(ilut1e_1.get_L(), u_dbl));
        EXPECT_LE(count_zeros(ilut1e_1.get_L(), u_dbl),
                  count_zeros(ilut1e_0.get_L(), u_dbl));

        cout << count_zeros(ilut1e_4.get_U(), u_dbl) << " "
             << count_zeros(ilut1e_3.get_U(), u_dbl) << " "
             << count_zeros(ilut1e_2.get_U(), u_dbl) << " "
             << count_zeros(ilut1e_1.get_U(), u_dbl) << " "
             << count_zeros(ilut1e_0.get_U(), u_dbl) << endl;

        cout << count_zeros(ilut1e_4.get_L(), u_dbl) << " "
             << count_zeros(ilut1e_3.get_L(), u_dbl) << " "
             << count_zeros(ilut1e_2.get_L(), u_dbl) << " "
             << count_zeros(ilut1e_1.get_L(), u_dbl) << " "
             << count_zeros(ilut1e_0.get_L(), u_dbl) << endl;

    }

    template<template <typename> typename M>
    void TestILUTDroppingLimits(bool pivot) {

        // Test that max dropping just gives the diagonal since everything else gets dropped
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));
        ILU<M, double> ilu_all_drop(A, DBL_MAX, n, u_dbl, pivot);

        ASSERT_MATRIX_LOWTRI(ilu_all_drop.get_L(), u_dbl);
        ASSERT_MATRIX_LOWTRI(ilu_all_drop.get_U(), u_dbl);
        ASSERT_MATRIX_UPPTRI(ilu_all_drop.get_L(), u_dbl);
        ASSERT_MATRIX_UPPTRI(ilu_all_drop.get_U(), u_dbl);

        for (int i=0; i<n; ++i) {
            ASSERT_NEAR(ilu_all_drop.get_L().coeff(i, i), 1., u_dbl);
            ASSERT_NEAR(ilu_all_drop.get_U().coeff(i, i), A.coeff(i, i), u_dbl);
        }

    }

    template<template <typename> typename M>
    void TestKeepPLargest(bool pivot) {

        // Test that 0 dropping leads to just p largest being retained
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu_keep_8(A, 0., 8, u_dbl, pivot);
        ILU<M, double> ilu_keep_7(A, 0., 7, u_dbl, pivot);
        ILU<M, double> ilu_keep_6(A, 0., 6, u_dbl, pivot);
        ILU<M, double> ilu_keep_5(A, 0., 5, u_dbl, pivot);
        ILU<M, double> ilu_keep_4(A, 0., 4, u_dbl, pivot);
        ILU<M, double> ilu_keep_3(A, 0., 3, u_dbl, pivot);
        ILU<M, double> ilu_keep_2(A, 0., 2, u_dbl, pivot);
        ILU<M, double> ilu_keep_1(A, 0., 1, u_dbl, pivot);
        ILU<M, double> ilu_keep_0(A, 0., 0, u_dbl, pivot);

        for (int i=0; i<n; ++i) {
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_8.get_U().col(i)), u_dbl), 8+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_7.get_U().col(i)), u_dbl), 7+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_6.get_U().col(i)), u_dbl), 6+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_5.get_U().col(i)), u_dbl), 5+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_4.get_U().col(i)), u_dbl), 4+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_3.get_U().col(i)), u_dbl), 3+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_2.get_U().col(i)), u_dbl), 2+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_1.get_U().col(i)), u_dbl), 1+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_0.get_U().col(i)), u_dbl), 0+1);
        }

        for (int i=0; i<n; ++i) {
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_8.get_L().col(i)), u_dbl), 8+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_7.get_L().col(i)), u_dbl), 7+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_6.get_L().col(i)), u_dbl), 6+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_5.get_L().col(i)), u_dbl), 5+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_4.get_L().col(i)), u_dbl), 4+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_3.get_L().col(i)), u_dbl), 3+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_2.get_L().col(i)), u_dbl), 2+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_1.get_L().col(i)), u_dbl), 1+1);
            ASSERT_LE(n-count_zeros(MatrixVector<double>(ilu_keep_0.get_L().col(i)), u_dbl), 0+1);
        }

    }

};

TEST_F(ILUTP_Test, TestEquivalentILUTNoDropAndDenseILU0_Dense) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixDense>();
}
TEST_F(ILUTP_Test, TestEquivalentILUTNoDropAndDenseILU0_Sparse) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixSparse>();
}

TEST_F(ILUTP_Test, TestILUTDropping_Dense) { TestILUTDropping<MatrixDense>(false); }
TEST_F(ILUTP_Test, TestILUTDropping_Sparse) { TestILUTDropping<MatrixSparse>(false); }

TEST_F(ILUTP_Test, TestILUTDropping_Pivoted_Dense) { TestILUTDropping<MatrixDense>(true); }
TEST_F(ILUTP_Test, TestILUTDropping_Pivoted_Sparse) { TestILUTDropping<MatrixSparse>(true); }

TEST_F(ILUTP_Test, TestILUTDroppingLimits_Dense) { TestILUTDroppingLimits<MatrixDense>(false); }
TEST_F(ILUTP_Test, TestILUTDroppingLimits_Sparse) { TestILUTDroppingLimits<MatrixSparse>(false); }

TEST_F(ILUTP_Test, TestILUTDroppingLimits_Pivoted_Dense) { TestILUTDroppingLimits<MatrixDense>(true); }
TEST_F(ILUTP_Test, TestILUTDroppingLimits_Pivoted_Sparse) { TestILUTDroppingLimits<MatrixSparse>(true); }

TEST_F(ILUTP_Test, TestKeepPLargest_Dense) { TestKeepPLargest<MatrixDense>(false); }
TEST_F(ILUTP_Test, TestKeepPLargest_Sparse) { TestKeepPLargest<MatrixSparse>(false); }

TEST_F(ILUTP_Test, TestKeepPLargest_Pivoted_Dense) { TestKeepPLargest<MatrixDense>(true); }
TEST_F(ILUTP_Test, TestKeepPLargest_Pivoted_Sparse) { TestKeepPLargest<MatrixSparse>(true); }


