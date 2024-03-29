#include "../test.h"

#include "preconditioners/implemented_preconditioners.h"

class ILUTP_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestEquivalentILUTNoDropAndDenseILU0() {

        // Test ILU(0) and ILUT(0, n) [No Dropping] Give the same dense decomp
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu0(A, Tol<double>::roundoff(), false);
        ILUPreconditioner<M, double> ilut0(A, 0., n, Tol<double>::roundoff(), false);

        ASSERT_MATRIX_EQ(ilu0.get_L(), ilut0.get_L());
        ASSERT_MATRIX_EQ(ilu0.get_U(), ilut0.get_U());

    }

    template<template <typename> typename M>
    void TestILUTDropping(bool pivot) {

        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_sparse_A.csv"))
        );

        // Check multiple rising thresholds ensuring that each ilu is closer to the matrix and that
        // all have correct form for L and U
        ILUPreconditioner<M, double> ilut1e_4(A, 1e-4, n, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilut1e_3(A, 1e-3, n, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilut1e_2(A, 1e-2, n, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilut1e_1(A, 1e-1, n, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilut1e_0(A, 1e-0, n, Tol<double>::roundoff(), pivot);

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
        EXPECT_LE((A-ilut1e_4.get_L()*ilut1e_4.get_U()).norm().get_scalar(),
                  (A-ilut1e_3.get_L()*ilut1e_3.get_U()).norm().get_scalar());
        EXPECT_LE((A-ilut1e_3.get_L()*ilut1e_3.get_U()).norm().get_scalar(),
                  (A-ilut1e_2.get_L()*ilut1e_2.get_U()).norm().get_scalar());
        EXPECT_LE((A-ilut1e_2.get_L()*ilut1e_2.get_U()).norm().get_scalar(),
                  (A-ilut1e_1.get_L()*ilut1e_1.get_U()).norm().get_scalar());
        EXPECT_LE((A-ilut1e_1.get_L()*ilut1e_1.get_U()).norm().get_scalar(),
                  (A-ilut1e_0.get_L()*ilut1e_0.get_U()).norm().get_scalar());

        // Test that each higher threshold has more zeros
        EXPECT_LE(count_zeros(ilut1e_4.get_U(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_3.get_U(), Tol<double>::roundoff()));
        EXPECT_LE(count_zeros(ilut1e_3.get_U(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_2.get_U(), Tol<double>::roundoff()));
        EXPECT_LE(count_zeros(ilut1e_2.get_U(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_1.get_U(), Tol<double>::roundoff()));
        EXPECT_LE(count_zeros(ilut1e_1.get_U(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_0.get_U(), Tol<double>::roundoff()));

        EXPECT_LE(count_zeros(ilut1e_4.get_L(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_3.get_L(), Tol<double>::roundoff()));
        EXPECT_LE(count_zeros(ilut1e_3.get_L(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_2.get_L(), Tol<double>::roundoff()));
        EXPECT_LE(count_zeros(ilut1e_2.get_L(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_1.get_L(), Tol<double>::roundoff()));
        EXPECT_LE(count_zeros(ilut1e_1.get_L(), Tol<double>::roundoff()),
                  count_zeros(ilut1e_0.get_L(), Tol<double>::roundoff()));

        std::cout << count_zeros(ilut1e_4.get_U(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_3.get_U(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_2.get_U(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_1.get_U(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_0.get_U(), Tol<double>::roundoff()) << std::endl;

        std::cout << count_zeros(ilut1e_4.get_L(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_3.get_L(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_2.get_L(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_1.get_L(), Tol<double>::roundoff()) << " "
                  << count_zeros(ilut1e_0.get_L(), Tol<double>::roundoff()) << std::endl;

    }

    template<template <typename> typename M>
    void TestILUTDroppingLimits(bool pivot) {

        // Test that max dropping just gives the diagonal since everything else gets dropped
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_sparse_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_all_drop(A, DBL_MAX, n, Tol<double>::roundoff(), pivot);

        ASSERT_MATRIX_LOWTRI(ilu_all_drop.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_LOWTRI(ilu_all_drop.get_U(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_all_drop.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_all_drop.get_U(), Tol<double>::roundoff());

        for (int i=0; i<n; ++i) {
            ASSERT_NEAR(
                ilu_all_drop.get_L().get_elem(i, i).get_scalar(),
                1.,
                Tol<double>::roundoff()
            );
            ASSERT_NEAR(
                ilu_all_drop.get_U().get_elem(i, i).get_scalar(),
                A.get_elem(i, i).get_scalar(),
                Tol<double>::roundoff()
            );
        }

    }

    template<template <typename> typename M>
    void TestKeepPLargest(bool pivot) {

        // Test that 0 dropping leads to just p largest being retained
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_keep_8(A, 0., 8, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_7(A, 0., 7, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_6(A, 0., 6, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_5(A, 0., 5, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_4(A, 0., 4, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_3(A, 0., 3, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_2(A, 0., 2, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_1(A, 0., 1, Tol<double>::roundoff(), pivot);
        ILUPreconditioner<M, double> ilu_keep_0(A, 0., 0, Tol<double>::roundoff(), pivot);

        for (int i=0; i<n; ++i) {
            ASSERT_LE(
                n-count_zeros(ilu_keep_8.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                8+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_7.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                7+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_6.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                6+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_5.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                5+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_4.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                4+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_3.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                3+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_2.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                2+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_1.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                1+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_0.get_U().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                0+1
            );
        }

        for (int i=0; i<n; ++i) {
            ASSERT_LE(
                n-count_zeros(ilu_keep_8.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                8+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_7.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                7+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_6.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                6+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_5.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                5+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_4.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                4+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_3.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                3+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_2.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                2+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_1.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                1+1
            );
            ASSERT_LE(
                n-count_zeros(ilu_keep_0.get_L().get_col(i).copy_to_vec(), Tol<double>::roundoff()),
                0+1
            );
        }

    }

};

TEST_F(ILUTP_Test, TestEquivalentILUTNoDropAndDenseILU0) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixDense>();
    // TestEquivalentILUTNoDropAndDenseILU0<MatrixSparse>();
}

TEST_F(ILUTP_Test, TestILUTDropping) {
    TestILUTDropping<MatrixDense>(false);
    // TestILUTDropping<MatrixSparse>(false);
}

TEST_F(ILUTP_Test, TestILUTDropping_Pivoted) {
    TestILUTDropping<MatrixDense>(true);
    // TestILUTDropping<MatrixSparse>(true);
}

TEST_F(ILUTP_Test, TestILUTDroppingLimits) {
    TestILUTDroppingLimits<MatrixDense>(false);
    // TestILUTDroppingLimits<MatrixSparse>(false);
}

TEST_F(ILUTP_Test, TestILUTDroppingLimits_Pivoted) {
    TestILUTDroppingLimits<MatrixDense>(true);
    // TestILUTDroppingLimits<MatrixSparse>(true);
}

TEST_F(ILUTP_Test, TestKeepPLargest) {
    TestKeepPLargest<MatrixDense>(false);
    // TestKeepPLargest<MatrixSparse>(false);
}

TEST_F(ILUTP_Test, TestKeepPLargest_Pivoted) {
    TestKeepPLargest<MatrixDense>(true);
    // TestKeepPLargest<MatrixSparse>(true);
}