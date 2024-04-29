#include "../test.h"

#include <cmath>

#include "preconditioners/implemented_preconditioners.h"

class ILUTP_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestEquivalentILUTNoDropAndDenseILU0() {

        // Test ILU(0) and ILUT(0, n) [No Dropping] Give the same dense decomp
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu0(A, false);
        ILUPreconditioner<M, double> ilut0(A, 0., n, false);
        ASSERT_MATRIX_EQ(ilut0.get_U(), ilu0.get_U());

    }

    template <template <typename> typename M>
    void TestILUTDropping(bool pivot) {

        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_A.csv"))
        );

        // Check multiple rising thresholds ensuring that each ilu is closer to the matrix and that
        // all have correct form for L and U
        ILUPreconditioner<M, double> ilut1e_4(A, 1e-4, n, pivot);
        ILUPreconditioner<M, double> ilut1e_3(A, 1e-3, n, pivot);
        ILUPreconditioner<M, double> ilut1e_2(A, 1e-2, n, pivot);
        ILUPreconditioner<M, double> ilut1e_1(A, 1e-1, n, pivot);
        ILUPreconditioner<M, double> ilut1e_0(A, 1e-0, n, pivot);

        ASSERT_MATRIX_LOWTRI(ilut1e_4.get_L(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_LOWTRI(ilut1e_3.get_L(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_LOWTRI(ilut1e_2.get_L(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_LOWTRI(ilut1e_1.get_L(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_LOWTRI(ilut1e_0.get_L(), Tol<double>::dbl_ilu_elem_tol());

        ASSERT_MATRIX_UPPTRI(ilut1e_4.get_U(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_UPPTRI(ilut1e_3.get_U(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_UPPTRI(ilut1e_2.get_U(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_UPPTRI(ilut1e_1.get_U(), Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_UPPTRI(ilut1e_0.get_U(), Tol<double>::dbl_ilu_elem_tol());

        // Test that each lower threshold is better than the higher one w.r.t
        // Frobenius norm
        EXPECT_LE(
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_4.get_L())*MatrixDense<double>(ilut1e_4.get_U())
            ).norm().get_scalar(),
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_3.get_L())*MatrixDense<double>(ilut1e_3.get_U())
            ).norm().get_scalar()
        );
        EXPECT_LE(
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_3.get_L())*MatrixDense<double>(ilut1e_3.get_U())
            ).norm().get_scalar(),
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_2.get_L())*MatrixDense<double>(ilut1e_2.get_U())
            ).norm().get_scalar()
        );
        EXPECT_LE(
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_2.get_L())*MatrixDense<double>(ilut1e_2.get_U())
            ).norm().get_scalar(),
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_1.get_L())*MatrixDense<double>(ilut1e_1.get_U())
            ).norm().get_scalar()
        );
        EXPECT_LE(
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_1.get_L())*MatrixDense<double>(ilut1e_1.get_U())
            ).norm().get_scalar(),
            (MatrixDense<double>(A)-
             MatrixDense<double>(ilut1e_0.get_L())*MatrixDense<double>(ilut1e_0.get_U())
            ).norm().get_scalar()
        );

        // Test that each smaller threshold has more non zeroes
        EXPECT_GE(ilut1e_4.get_U().non_zeros(), ilut1e_3.get_U().non_zeros());
        EXPECT_GE(ilut1e_3.get_U().non_zeros(), ilut1e_2.get_U().non_zeros());
        EXPECT_GE(ilut1e_2.get_U().non_zeros(), ilut1e_1.get_U().non_zeros());
        EXPECT_GE(ilut1e_1.get_U().non_zeros(), ilut1e_0.get_U().non_zeros());

        EXPECT_GE(ilut1e_4.get_L().non_zeros(), ilut1e_3.get_L().non_zeros());
        EXPECT_GE(ilut1e_3.get_L().non_zeros(), ilut1e_2.get_L().non_zeros());
        EXPECT_GE(ilut1e_2.get_L().non_zeros(), ilut1e_1.get_L().non_zeros());
        EXPECT_GE(ilut1e_1.get_L().non_zeros(), ilut1e_0.get_L().non_zeros());

        std::cout << ilut1e_4.get_U().non_zeros() << " "
                  << ilut1e_3.get_U().non_zeros() << " "
                  << ilut1e_2.get_U().non_zeros() << " "
                  << ilut1e_1.get_U().non_zeros() << " "
                  << ilut1e_0.get_U().non_zeros() << std::endl;

        std::cout << ilut1e_4.get_L().non_zeros() << " "
                  << ilut1e_3.get_L().non_zeros() << " "
                  << ilut1e_2.get_L().non_zeros() << " "
                  << ilut1e_1.get_L().non_zeros() << " "
                  << ilut1e_0.get_L().non_zeros() << std::endl;

    }

    template<template <typename> typename M>
    void TestILUTDroppingLimits(bool pivot) {

        // Test that max dropping just gives the diagonal since everything else gets dropped
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_all_drop(A, DBL_MAX, n, pivot);

        ASSERT_MATRIX_LOWTRI(ilu_all_drop.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_LOWTRI(ilu_all_drop.get_U(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_all_drop.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_all_drop.get_U(), Tol<double>::roundoff());

        for (int j=0; j<n; ++j) {
            ASSERT_NEAR(
                ilu_all_drop.get_L().get_elem(j, j).get_scalar(),
                1.,
                Tol<double>::roundoff()
            );
            if (pivot) {
                double max_mag_val = 0.;
                for (int i=0; i<n ; ++i) {
                    Scalar<double> a_ij = A.get_elem(i, j);
                    Scalar<double> a_ij_abs = A.get_elem(i, j);
                    a_ij_abs.abs();
                    if (a_ij_abs.get_scalar() > std::abs(max_mag_val)) {
                        max_mag_val = a_ij.get_scalar();
                    }
                }
                ASSERT_NEAR(
                    ilu_all_drop.get_U().get_elem(j, j).get_scalar(),
                    max_mag_val,
                    Tol<double>::roundoff()
                );
            } else {
                ASSERT_NEAR(
                    ilu_all_drop.get_U().get_elem(j, j).get_scalar(),
                    A.get_elem(j, j).get_scalar(),
                    Tol<double>::roundoff()
                );
            }
        }

    }

    template <template <typename> typename M>
    void TestKeepPLargest(bool pivot) {

        // Test that 0 tau (not applying drop rule tau) and p entries just leads to p largest being retained
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_keep_8(A, 0., 8, pivot);
        ILUPreconditioner<M, double> ilu_keep_7(A, 0., 7, pivot);
        ILUPreconditioner<M, double> ilu_keep_6(A, 0., 6, pivot);
        ILUPreconditioner<M, double> ilu_keep_5(A, 0., 5, pivot);
        ILUPreconditioner<M, double> ilu_keep_4(A, 0., 4, pivot);
        ILUPreconditioner<M, double> ilu_keep_3(A, 0., 3, pivot);
        ILUPreconditioner<M, double> ilu_keep_2(A, 0., 2, pivot);
        ILUPreconditioner<M, double> ilu_keep_1(A, 0., 1, pivot);

        for (int j=0; j<n; ++j) {
            ASSERT_LE(ilu_keep_8.get_U().get_col(j).copy_to_vec().non_zeros(), 8);
            ASSERT_LE(ilu_keep_7.get_U().get_col(j).copy_to_vec().non_zeros(), 7);
            ASSERT_LE(ilu_keep_6.get_U().get_col(j).copy_to_vec().non_zeros(), 6);
            ASSERT_LE(ilu_keep_5.get_U().get_col(j).copy_to_vec().non_zeros(), 5);
            ASSERT_LE(ilu_keep_4.get_U().get_col(j).copy_to_vec().non_zeros(), 4);
            ASSERT_LE(ilu_keep_3.get_U().get_col(j).copy_to_vec().non_zeros(), 3);
            ASSERT_LE(ilu_keep_2.get_U().get_col(j).copy_to_vec().non_zeros(), 2);
            ASSERT_LE(ilu_keep_1.get_U().get_col(j).copy_to_vec().non_zeros(), 1);
        }

        for (int j=0; j<n; ++j) {
            ASSERT_LE(ilu_keep_8.get_L().get_col(j).copy_to_vec().non_zeros(), 8);
            ASSERT_LE(ilu_keep_7.get_L().get_col(j).copy_to_vec().non_zeros(), 7);
            ASSERT_LE(ilu_keep_6.get_L().get_col(j).copy_to_vec().non_zeros(), 6);
            ASSERT_LE(ilu_keep_5.get_L().get_col(j).copy_to_vec().non_zeros(), 5);
            ASSERT_LE(ilu_keep_4.get_L().get_col(j).copy_to_vec().non_zeros(), 4);
            ASSERT_LE(ilu_keep_3.get_L().get_col(j).copy_to_vec().non_zeros(), 3);
            ASSERT_LE(ilu_keep_2.get_L().get_col(j).copy_to_vec().non_zeros(), 2);
            ASSERT_LE(ilu_keep_1.get_L().get_col(j).copy_to_vec().non_zeros(), 1);
        }

    }

};

TEST_F(ILUTP_Test, TestEquivalentILUTNoDropAndDenseILU0) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixDense>();
    TestEquivalentILUTNoDropAndDenseILU0<NoFillMatrixSparse>();
}

TEST_F(ILUTP_Test, TestILUTDropping) {
    TestILUTDropping<MatrixDense>(false);
    TestILUTDropping<NoFillMatrixSparse>(false);
}

TEST_F(ILUTP_Test, TestILUTDropping_Pivoted) {
    TestILUTDropping<MatrixDense>(true);
    TestILUTDropping<NoFillMatrixSparse>(true);
}

TEST_F(ILUTP_Test, TestILUTDroppingLimits) {
    TestILUTDroppingLimits<MatrixDense>(false);
    TestILUTDroppingLimits<NoFillMatrixSparse>(false);
}

TEST_F(ILUTP_Test, TestILUTDroppingLimits_Pivoted) {
    TestILUTDroppingLimits<MatrixDense>(true);
    TestILUTDroppingLimits<NoFillMatrixSparse>(true);
}

TEST_F(ILUTP_Test, TestKeepPLargest) {
    TestKeepPLargest<MatrixDense>(false);
    TestKeepPLargest<NoFillMatrixSparse>(false);
}

TEST_F(ILUTP_Test, TestKeepPLargest_Pivoted) {
    TestKeepPLargest<MatrixDense>(true);
    TestKeepPLargest<NoFillMatrixSparse>(true);
}