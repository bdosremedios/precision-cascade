#include "test_Matrix.h"

#include "types/MatrixSparse/ImmutableMatrixSparse.h"

class ImmutableMatrixSparse_Test: public Matrix_Test<ImmutableMatrixSparse>
{
public:

    template <typename T>
    void TestCoeffJustGetAccess() {

        const int m(5);
        const int n(4);

        // Test simple counting up
        ImmutableMatrixSparse<T> test_mat_1(
            TestBase::bundle,
            {{static_cast<T>(1.), static_cast<T>(2.), static_cast<T>(3.), static_cast<T>(4.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(7.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(11.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(15.), static_cast<T>(16.)},
             {static_cast<T>(17.), static_cast<T>(18.), static_cast<T>(19.), static_cast<T>(20.)}}
        );

        T test_elem = static_cast<T>(1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat_1.get_elem(i, j).get_scalar(), test_elem);
                test_elem += static_cast<T>(1);
            }
        }

        // Modify first row to be all -1.
        ImmutableMatrixSparse<T> test_mat_2(
            TestBase::bundle,
            {{static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(7.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(11.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(15.), static_cast<T>(16.)},
             {static_cast<T>(17.), static_cast<T>(18.), static_cast<T>(19.), static_cast<T>(20.)}}
        );

        test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(test_mat_2.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else {
                    ASSERT_EQ(test_mat_2.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 4 row as decreasing by -1 from -1
        ImmutableMatrixSparse<T> test_mat_3(
            TestBase::bundle,
            {{static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(7.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(11.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(15.), static_cast<T>(16.)},
             {static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-3.), static_cast<T>(-4.)}}
        );

        test_elem = static_cast<T>(1);
        T row_5_test_elem = static_cast<T>(-1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(test_mat_3.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else if (i == 4) {
                    ASSERT_EQ(test_mat_3.get_elem(i, j).get_scalar(), row_5_test_elem);
                    row_5_test_elem += static_cast<T>(-1.);
                } else {
                    ASSERT_EQ(test_mat_3.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 2 col as increasing by 1 from -5
        ImmutableMatrixSparse<T> test_mat_4(
            TestBase::bundle,
            {{static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-5.), static_cast<T>(-1.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(-4.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(-3.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(-2.), static_cast<T>(16.)},
             {static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-1.), static_cast<T>(-4.)}}
        );

        test_elem = static_cast<T>(1);
        row_5_test_elem = static_cast<T>(-1.);
        T coL_3_test_elem = static_cast<T>(-5.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (j == 2) {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), coL_3_test_elem);
                    coL_3_test_elem += static_cast<T>(1.);
                } else if (i == 0) {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else if (i == 4) {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), row_5_test_elem);
                    row_5_test_elem += static_cast<T>(-1.);
                    if (j == 1) { row_5_test_elem += static_cast<T>(-1.); }
                } else {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Test reading an all zero matrix
        ImmutableMatrixSparse<T> test_mat_5(
            TestBase::bundle,
            {{static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)}}
        );
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat_5.get_elem(i, j).get_scalar(), static_cast<T>(0.));
            }
        }

        // Test reading a sparse matrix
        ImmutableMatrixSparse<T> test_mat_6(
            TestBase::bundle,
            {{static_cast<T>(1.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(4.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(-5.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)}}
        );
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if ((i == 0) && (j == 0)) {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(1.));
                } else if ((i == 1) && (j == 3)) {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(4.));
                } else if ((i == 3) && (j == 1)) {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(-5.));
                } else {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(0.));
                }
            }
        }
    
    }

    void TestBadCoeffJustGetAccess() {
        
        constexpr int m(24);
        constexpr int n(12);
        ImmutableMatrixSparse<double> test_mat(TestBase::bundle, m, n);
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(0, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(0, n); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(-1, 0); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(m, 0); });

    }

    template <typename T>
    void TestDynamicMemConstruction() {
    
        const int m_manual(2);
        const int n_manual(3);
        const int nnz_manual(4);

        int h_col_offsets[3] = {0, 2, 3};
        int h_row_indices[4] = {0, 1, 1, 0};
        T h_vals[4] = {
            static_cast<T>(-5), static_cast<T>(3), static_cast<T>(3.5), static_cast<T>(-20)
        };

        ImmutableMatrixSparse<T> mat(
            TestBase::bundle,
            h_col_offsets, h_row_indices, h_vals,
            m_manual, n_manual, nnz_manual
        );

        ASSERT_EQ(mat.rows(), m_manual);
        ASSERT_EQ(mat.cols(), n_manual);
        ASSERT_EQ(mat.non_zeros(), nnz_manual);

        ASSERT_EQ(mat.get_elem(0, 0).get_scalar(), static_cast<T>(-5));
        ASSERT_EQ(mat.get_elem(1, 0).get_scalar(), static_cast<T>(3));
        ASSERT_EQ(mat.get_elem(0, 1).get_scalar(), static_cast<T>(0));
        ASSERT_EQ(mat.get_elem(1, 1).get_scalar(), static_cast<T>(3.5));
        ASSERT_EQ(mat.get_elem(0, 2).get_scalar(), static_cast<T>(-20));
        ASSERT_EQ(mat.get_elem(1, 2).get_scalar(), static_cast<T>(0));
    
    }

    template <typename T>
    void TestDynamicMemCopyToPtr() {
    
        const int m_manual(2);
        const int n_manual(3);
        const int nnz_manual(4);

        ImmutableMatrixSparse<T> mat_manual(
            TestBase::bundle,
            {{static_cast<T>(-5), static_cast<T>(0.), static_cast<T>(-20)},
             {static_cast<T>(3), static_cast<T>(3.5), static_cast<T>(0.)}}
        );

        int *h_col_offsets = static_cast<int *>(malloc(n_manual*sizeof(int)));
        int *h_row_indices = static_cast<int *>(malloc(nnz_manual*sizeof(int)));
        T *h_vals = static_cast<T *>(malloc(nnz_manual*sizeof(T)));

        mat_manual.copy_data_to_ptr(
            h_col_offsets, h_row_indices, h_vals,
            m_manual, n_manual, nnz_manual
        );

        ASSERT_EQ(h_col_offsets[0], 0);
        ASSERT_EQ(h_col_offsets[1], 2);
        ASSERT_EQ(h_col_offsets[2], 3);

        ASSERT_EQ(h_row_indices[0], 0);
        ASSERT_EQ(h_row_indices[1], 1);
        ASSERT_EQ(h_row_indices[2], 1);
        ASSERT_EQ(h_row_indices[3], 0);

        ASSERT_EQ(h_vals[0], static_cast<T>(-5));
        ASSERT_EQ(h_vals[1], static_cast<T>(3));
        ASSERT_EQ(h_vals[2], static_cast<T>(3.5));
        ASSERT_EQ(h_vals[3], static_cast<T>(-20));

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);
    
        const int m_manual_2(3);
        const int n_manual_2(5);
        const int nnz_manual_2(8);

        ImmutableMatrixSparse<T> mat_manual_2(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
              static_cast<T>(0), static_cast<T>(5)},
             {static_cast<T>(6), static_cast<T>(0), static_cast<T>(0),
              static_cast<T>(9), static_cast<T>(10)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
              static_cast<T>(14), static_cast<T>(0)}}
        );

        int *h_col_offsets_2 = static_cast<int *>(malloc(n_manual_2*sizeof(int)));
        int *h_row_indices_2 = static_cast<int *>(malloc(nnz_manual_2*sizeof(int)));
        T *h_vals_2 = static_cast<T *>(malloc(nnz_manual_2*sizeof(T)));

        mat_manual_2.copy_data_to_ptr(
            h_col_offsets_2, h_row_indices_2, h_vals_2,
            m_manual_2, n_manual_2, nnz_manual_2
        );

        ASSERT_EQ(h_col_offsets_2[0], 0);
        ASSERT_EQ(h_col_offsets_2[1], 2);
        ASSERT_EQ(h_col_offsets_2[2], 3);
        ASSERT_EQ(h_col_offsets_2[3], 4);
        ASSERT_EQ(h_col_offsets_2[4], 6);

        ASSERT_EQ(h_row_indices_2[0], 0);
        ASSERT_EQ(h_row_indices_2[1], 1);
        ASSERT_EQ(h_row_indices_2[2], 0);
        ASSERT_EQ(h_row_indices_2[3], 0);
        ASSERT_EQ(h_row_indices_2[4], 1);
        ASSERT_EQ(h_row_indices_2[5], 2);
        ASSERT_EQ(h_row_indices_2[6], 0);
        ASSERT_EQ(h_row_indices_2[7], 1);

        ASSERT_EQ(h_vals_2[0], static_cast<T>(1));
        ASSERT_EQ(h_vals_2[1], static_cast<T>(6));
        ASSERT_EQ(h_vals_2[2], static_cast<T>(2));
        ASSERT_EQ(h_vals_2[3], static_cast<T>(3));
        ASSERT_EQ(h_vals_2[4], static_cast<T>(9));
        ASSERT_EQ(h_vals_2[5], static_cast<T>(14));
        ASSERT_EQ(h_vals_2[6], static_cast<T>(5));
        ASSERT_EQ(h_vals_2[7], static_cast<T>(10));

        free(h_col_offsets_2);
        free(h_row_indices_2);
        free(h_vals_2);

    }

    void TestBadDynamicMemCopyToPtr() {

        const int m_manual(4);
        const int n_manual(5);
        const int nnz_manual(6);
        ImmutableMatrixSparse<double> mat(
            TestBase::bundle,
            {{1., 0., 0., 0., 0.},
             {0., 3.5, 20., 0., 0.},
             {0., 2., 0., 0., 0.},
             {-1., 0., 0., 0., -0.5}}
        );

        int *h_col_offsets = static_cast<int *>(malloc(n_manual*sizeof(int)));
        int *h_row_indices = static_cast<int *>(malloc(nnz_manual*sizeof(int)));
        double *h_vals = static_cast<double *>(malloc(nnz_manual*sizeof(double)));
        
        auto try_row_too_small = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual-2, n_manual, nnz_manual
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_small);

        auto try_row_too_large = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual+2, n_manual, nnz_manual
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_large);

        auto try_col_too_small = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual, n_manual-2, nnz_manual
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_col_too_small);

        auto try_col_too_large = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual, n_manual+2, nnz_manual
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_col_too_large);

        auto try_nnz_too_small = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual, n_manual, nnz_manual-2
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_nnz_too_small);

        auto try_nnz_too_large = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual, n_manual, nnz_manual+2
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_nnz_too_large);

        auto try_match_wrong_dim_row = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                n_manual, n_manual, nnz_manual
            ); 
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_match_wrong_dim_row);

        auto try_match_wrong_dim_col = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual, nnz_manual, nnz_manual
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_match_wrong_dim_col);

        auto try_match_wrong_dim_nnz = [&]() {
            mat.copy_data_to_ptr(
                h_col_offsets, h_row_indices, h_vals,
                m_manual, n_manual, m_manual
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_match_wrong_dim_nnz);

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

    }

    template <typename T>
    void TestRandomMatrixCreation() {

        constexpr int m_rand(30);
        constexpr int n_rand(40);
        constexpr double miss_tol(0.05);

        // Check that zero fill_prob gives empty matrix
        ImmutableMatrixSparse<T> test_rand_empty(
            ImmutableMatrixSparse<T>::Random(TestBase::bundle, m_rand, n_rand, 0.)
        );
        ASSERT_EQ(test_rand_empty.rows(), m_rand);
        ASSERT_EQ(test_rand_empty.cols(), n_rand);
        ASSERT_EQ(test_rand_empty.non_zeros(), 0);

        // Just test for non-zero fill_prob gives right size and numbers aren't generally the same
        // (check middle numbers are different from 5 adjacent above and below or all are equally zero)
        // also check if fill probability is right with 5% error
        ImmutableMatrixSparse<T> test_rand_full(
            ImmutableMatrixSparse<T>::Random(TestBase::bundle, m_rand, n_rand, 1.)
        );
        ASSERT_EQ(test_rand_full.rows(), m_rand);
        ASSERT_EQ(test_rand_full.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    (((test_rand_full.get_elem(i, j).get_scalar() !=
                       test_rand_full.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_full.get_elem(i, j).get_scalar() !=
                       test_rand_full.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_full.get_elem(i, j).get_scalar() !=
                       test_rand_full.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_full.get_elem(i, j).get_scalar() !=
                       test_rand_full.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_full.get_elem(i, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_full.get_elem(i-1, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_full.get_elem(i+1, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_full.get_elem(i, j-1).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_full.get_elem(i, j+1).get_scalar() == static_cast<T>(0.))))
                );
            }
        }
        ASSERT_NEAR(
            (static_cast<T>(test_rand_full.non_zeros())/
             static_cast<T>(test_rand_full.rows()*test_rand_full.cols())),
            1.,
            miss_tol
        );

        ImmutableMatrixSparse<T> test_rand_67(
            ImmutableMatrixSparse<T>::Random(TestBase::bundle, m_rand, n_rand, 0.67)
        );
        ASSERT_EQ(test_rand_67.rows(), m_rand);
        ASSERT_EQ(test_rand_67.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    (((test_rand_67.get_elem(i, j).get_scalar() != test_rand_67.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() != test_rand_67.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() != test_rand_67.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() != test_rand_67.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_67.get_elem(i, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_67.get_elem(i-1, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_67.get_elem(i+1, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_67.get_elem(i, j-1).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_67.get_elem(i, j+1).get_scalar() == static_cast<T>(0.))))
                );
            }
        }
        ASSERT_NEAR(
            (static_cast<T>(test_rand_67.non_zeros())/
             static_cast<T>(test_rand_67.rows()*test_rand_67.cols())),
            0.67,
            miss_tol
        );

        ImmutableMatrixSparse<T> test_rand_50(
            ImmutableMatrixSparse<T>::Random(TestBase::bundle, m_rand, n_rand, 0.5)
        );
        ASSERT_EQ(test_rand_50.rows(), m_rand);
        ASSERT_EQ(test_rand_50.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    (((test_rand_50.get_elem(i, j).get_scalar() != test_rand_50.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() != test_rand_50.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() != test_rand_50.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() != test_rand_50.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_50.get_elem(i, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_50.get_elem(i-1, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_50.get_elem(i+1, j).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_50.get_elem(i, j-1).get_scalar() == static_cast<T>(0.)) &&
                      (test_rand_50.get_elem(i, j+1).get_scalar() == static_cast<T>(0.))))
                );
            }
        }
        ASSERT_NEAR(
            (static_cast<T>(test_rand_50.non_zeros())/
             static_cast<T>(test_rand_50.rows()*test_rand_50.cols())),
            0.5,
            miss_tol
        );

    }

    // template <typename T>
    // void TestCol() { TestCol_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestTranspose() { TestTranspose_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestScale() { TestScale_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestMatVec() { TestMatVec_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestMatMat() { TestMatMat_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestNorm() { TestNorm_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestAddSub() { TestAddSub_Base<MatrixSparse, T>(); }

    // void TestCast() { TestCast_Base<MatrixSparse>(); }

};

TEST_F(ImmutableMatrixSparse_Test, TestPropertyAccess) {
    TestPropertyAccess<__half>();
    TestPropertyAccess<float>();
    TestPropertyAccess<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestNonZeros) {
    TestNonZeros<__half>();
    TestNonZeros<float>();
    TestNonZeros<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestPrintAndInfoString) {
    TestPrintAndInfoString();
}

TEST_F(ImmutableMatrixSparse_Test, TestConstruction) {
    TestConstruction<__half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestListInitialization) {
    TestListInitialization<__half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestBadListInitialization) {
    TestBadListInitialization();
}

TEST_F(ImmutableMatrixSparse_Test, TestCoeffJustGetAccess) {
    TestCoeffJustGetAccess<__half>();
    TestCoeffJustGetAccess<float>();
    TestCoeffJustGetAccess<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestBadCoeffJustGetAccess) {
    TestBadCoeffJustGetAccess();
}

TEST_F(ImmutableMatrixSparse_Test, TestCopyAssignment) {
    TestCopyAssignment<__half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestCopyConstructor) {
    TestCopyConstructor<__half>();
    TestCopyConstructor<float>();
    TestCopyConstructor<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestDynamicMemConstruction) {
    TestDynamicMemConstruction<__half>();
    TestDynamicMemConstruction<float>();
    TestDynamicMemConstruction<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestDynamicMemCopyToPtr) {
    TestDynamicMemCopyToPtr<__half>();
    TestDynamicMemCopyToPtr<float>();
    TestDynamicMemCopyToPtr<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestBadDynamicMemCopyToPtr) {
    TestBadDynamicMemCopyToPtr();
}
TEST_F(ImmutableMatrixSparse_Test, TestZeroMatrixCreation) {
    TestZeroMatrixCreation<__half>();
    TestZeroMatrixCreation<float>();
    TestZeroMatrixCreation<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestOnesMatrixCreation) {
    TestOnesMatrixCreation<__half>();
    TestOnesMatrixCreation<float>();
    TestOnesMatrixCreation<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestIdentityMatrixCreation) {
    TestIdentityMatrixCreation<__half>();
    TestIdentityMatrixCreation<float>();
    TestIdentityMatrixCreation<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestRandomMatrixCreation) {
    TestRandomMatrixCreation<__half>();
    TestRandomMatrixCreation<float>();
    TestRandomMatrixCreation<double>();
}

// TEST_F(ImmutableMatrixSparse_Test, TestCol) {
//     TestCol<__half>();
//     TestCol<float>();
//     TestCol<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestBadCol) {
//     TestBadCol();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestBlock) {
//     TestBlock<__half>();
//     TestBlock<float>();
//     TestBlock<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestBadBlock) {
//     TestBadBlock();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestScale) {
//     TestScale<__half>();
//     TestScale<float>();
//     TestScale<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestScaleAssignment) {
//     TestScaleAssignment<__half>();
//     TestScaleAssignment<float>();
//     TestScaleAssignment<double>();
// }

TEST_F(ImmutableMatrixSparse_Test, TestMaxMagElem) {
    TestMaxMagElem<__half>();
    TestMaxMagElem<float>();
    TestMaxMagElem<double>();
}

// TEST_F(ImmutableMatrixSparse_Test, TestNormalizeMagnitude) {
//     TestNormalizeMagnitude<__half>();
//     TestNormalizeMagnitude<float>();
//     TestNormalizeMagnitude<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestMatVec) {
//     TestMatVec<__half>();
//     TestMatVec<float>();
//     TestMatVec<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestBadMatVec) {
//     TestBadMatVec<__half>();
//     TestBadMatVec<float>();
//     TestBadMatVec<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestTransposeMatVec) {
//     TestTransposeMatVec<__half>();
//     TestTransposeMatVec<float>();
//     TestTransposeMatVec<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestBadTransposeMatVec) {
//     TestBadTransposeMatVec<__half>();
//     TestBadTransposeMatVec<float>();
//     TestBadTransposeMatVec<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestTranspose) {
//     TestTranspose<__half>();
//     TestTranspose<float>();
//     TestTranspose<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestMatMat) {
//     TestMatMat<__half>();
//     TestMatMat<float>();
//     TestMatMat<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestBadMatMat) {
//     TestBadMatMat<__half>();
//     TestBadMatMat<float>();
//     TestBadMatMat<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestAddSub) {
//     TestAddSub<__half>();
//     TestAddSub<float>();
//     TestAddSub<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestBadAddSub) {
//     TestBadAddSub<__half>();
//     TestBadAddSub<float>();
//     TestBadAddSub<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestNorm) {
//     TestNorm<__half>();
//     TestNorm<float>();
//     TestNorm<double>();
// }

// TEST_F(ImmutableMatrixSparse_Test, TestCast) { TestCast(); }

// TEST_F(ImmutableMatrixSparse_Test, TestBadCast) { TestBadCast(); }

// class MatrixSparse_Substitution_Test: public Matrix_Substitution_Test<MatrixSparse> {};

// TEST_F(MatrixSparse_Substitution_Test, TestBackwardSubstitution) {
//     TestBackwardSubstitution<__half>();
//     TestBackwardSubstitution<float>();
//     TestBackwardSubstitution<double>();
// }

// TEST_F(MatrixSparse_Substitution_Test, TestForwardSubstitution) {
//     TestForwardSubstitution<__half>();
//     TestForwardSubstitution<float>();
//     TestForwardSubstitution<double>();
// }