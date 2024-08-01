#include "test_Matrix.h"

#include "types/MatrixSparse/NoFillMatrixSparse.h"

class NoFillMatrixSparse_Test: public Matrix_Test<NoFillMatrixSparse>
{
public:

    template <typename TPrecision>
    void TestCoeffJustGetAccess() {

        const int m(5);
        const int n(4);

        // Test simple counting up
        NoFillMatrixSparse<TPrecision> test_mat_1(
            TestBase::bundle,
            {{static_cast<TPrecision>(1.), static_cast<TPrecision>(2.),
              static_cast<TPrecision>(3.), static_cast<TPrecision>(4.)},
             {static_cast<TPrecision>(5.), static_cast<TPrecision>(6.),
              static_cast<TPrecision>(7.), static_cast<TPrecision>(8.)},
             {static_cast<TPrecision>(9.), static_cast<TPrecision>(10.),
              static_cast<TPrecision>(11.), static_cast<TPrecision>(12.)},
             {static_cast<TPrecision>(13.), static_cast<TPrecision>(14.),
              static_cast<TPrecision>(15.), static_cast<TPrecision>(16.)},
             {static_cast<TPrecision>(17.), static_cast<TPrecision>(18.),
              static_cast<TPrecision>(19.), static_cast<TPrecision>(20.)}}
        );

        TPrecision test_elem = static_cast<TPrecision>(1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat_1.get_elem(i, j).get_scalar(), test_elem);
                test_elem += static_cast<TPrecision>(1);
            }
        }

        // Modify first row to be all -1.
        NoFillMatrixSparse<TPrecision> test_mat_2(
            TestBase::bundle,
            {{static_cast<TPrecision>(-1.), static_cast<TPrecision>(-1.),
              static_cast<TPrecision>(-1.), static_cast<TPrecision>(-1.)},
             {static_cast<TPrecision>(5.), static_cast<TPrecision>(6.),
              static_cast<TPrecision>(7.), static_cast<TPrecision>(8.)},
             {static_cast<TPrecision>(9.), static_cast<TPrecision>(10.),
              static_cast<TPrecision>(11.), static_cast<TPrecision>(12.)},
             {static_cast<TPrecision>(13.), static_cast<TPrecision>(14.),
              static_cast<TPrecision>(15.), static_cast<TPrecision>(16.)},
             {static_cast<TPrecision>(17.), static_cast<TPrecision>(18.),
              static_cast<TPrecision>(19.), static_cast<TPrecision>(20.)}}
        );

        test_elem = static_cast<TPrecision>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(
                        test_mat_2.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(-1.)
                    );
                } else {
                    ASSERT_EQ(
                        test_mat_2.get_elem(i, j).get_scalar(),
                        test_elem
                    );
                }
                test_elem += static_cast<TPrecision>(1);
            }
        }

        // Set index 4 row as decreasing by -1 from -1
        NoFillMatrixSparse<TPrecision> test_mat_3(
            TestBase::bundle,
            {{static_cast<TPrecision>(-1.), static_cast<TPrecision>(-1.),
              static_cast<TPrecision>(-1.), static_cast<TPrecision>(-1.)},
             {static_cast<TPrecision>(5.), static_cast<TPrecision>(6.),
              static_cast<TPrecision>(7.), static_cast<TPrecision>(8.)},
             {static_cast<TPrecision>(9.), static_cast<TPrecision>(10.),
              static_cast<TPrecision>(11.), static_cast<TPrecision>(12.)},
             {static_cast<TPrecision>(13.), static_cast<TPrecision>(14.),
              static_cast<TPrecision>(15.), static_cast<TPrecision>(16.)},
             {static_cast<TPrecision>(-1.), static_cast<TPrecision>(-2.),
              static_cast<TPrecision>(-3.), static_cast<TPrecision>(-4.)}}
        );

        test_elem = static_cast<TPrecision>(1);
        TPrecision row_5_test_elem = static_cast<TPrecision>(-1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(
                        test_mat_3.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(-1.)
                    );
                } else if (i == 4) {
                    ASSERT_EQ(
                        test_mat_3.get_elem(i, j).get_scalar(),
                        row_5_test_elem
                    );
                    row_5_test_elem += static_cast<TPrecision>(-1.);
                } else {
                    ASSERT_EQ(
                        test_mat_3.get_elem(i, j).get_scalar(),
                        test_elem
                    );
                }
                test_elem += static_cast<TPrecision>(1);
            }
        }

        // Set index 2 col as increasing by 1 from -5
        NoFillMatrixSparse<TPrecision> test_mat_4(
            TestBase::bundle,
            {{static_cast<TPrecision>(-1.), static_cast<TPrecision>(-1.),
              static_cast<TPrecision>(-5.), static_cast<TPrecision>(-1.)},
             {static_cast<TPrecision>(5.), static_cast<TPrecision>(6.),
              static_cast<TPrecision>(-4.), static_cast<TPrecision>(8.)},
             {static_cast<TPrecision>(9.), static_cast<TPrecision>(10.),
              static_cast<TPrecision>(-3.), static_cast<TPrecision>(12.)},
             {static_cast<TPrecision>(13.), static_cast<TPrecision>(14.),
              static_cast<TPrecision>(-2.), static_cast<TPrecision>(16.)},
             {static_cast<TPrecision>(-1.), static_cast<TPrecision>(-2.),
              static_cast<TPrecision>(-1.), static_cast<TPrecision>(-4.)}}
        );

        test_elem = static_cast<TPrecision>(1);
        row_5_test_elem = static_cast<TPrecision>(-1.);
        TPrecision coL_3_test_elem = static_cast<TPrecision>(-5.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (j == 2) {
                    ASSERT_EQ(
                        test_mat_4.get_elem(i, j).get_scalar(),
                        coL_3_test_elem
                    );
                    coL_3_test_elem += static_cast<TPrecision>(1.);
                } else if (i == 0) {
                    ASSERT_EQ(
                        test_mat_4.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(-1.)
                    );
                } else if (i == 4) {
                    ASSERT_EQ(
                        test_mat_4.get_elem(i, j).get_scalar(),
                        row_5_test_elem
                    );
                    row_5_test_elem += static_cast<TPrecision>(-1.);
                    if (j == 1) {
                        row_5_test_elem += static_cast<TPrecision>(-1.);
                    }
                } else {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<TPrecision>(1);
            }
        }

        // Test reading an all zero matrix
        NoFillMatrixSparse<TPrecision> test_mat_5(
            TestBase::bundle,
            {{static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)}}
        );
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(
                    test_mat_5.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

        // Test reading a sparse matrix
        NoFillMatrixSparse<TPrecision> test_mat_6(
            TestBase::bundle,
            {{static_cast<TPrecision>(1.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(4.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(-5.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)}}
        );
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if ((i == 0) && (j == 0)) {
                    ASSERT_EQ(
                        test_mat_6.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(1.)
                    );
                } else if ((i == 1) && (j == 3)) {
                    ASSERT_EQ(
                        test_mat_6.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(4.)
                    );
                } else if ((i == 3) && (j == 1)) {
                    ASSERT_EQ(
                        test_mat_6.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(-5.)
                    );
                } else {
                    ASSERT_EQ(
                        test_mat_6.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(0.)
                    );
                }
            }
        }
    
    }

    void TestBadCoeffJustGetAccess() {
        
        constexpr int m(24);
        constexpr int n(12);
        NoFillMatrixSparse<double> test_mat(TestBase::bundle, m, n);
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [&]() { test_mat.get_elem(0, -1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [&]() { test_mat.get_elem(0, n); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [&]() { test_mat.get_elem(-1, 0); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [&]() { test_mat.get_elem(m, 0); }
        );

    }

    template <typename TPrecision>
    void TestDynamicMemConstruction() {
    
        const int m_manual(2);
        const int n_manual(3);
        const int nnz_manual(4);

        int h_col_offsets[4] = {0, 2, 3, 4};
        int h_row_indices[4] = {0, 1, 1, 0};
        TPrecision h_vals[4] = {
            static_cast<TPrecision>(-5), static_cast<TPrecision>(3),
            static_cast<TPrecision>(3.5), static_cast<TPrecision>(-20)
        };

        NoFillMatrixSparse<TPrecision> mat(
            TestBase::bundle,
            h_col_offsets, h_row_indices, h_vals,
            m_manual, n_manual, nnz_manual
        );

        ASSERT_EQ(mat.rows(), m_manual);
        ASSERT_EQ(mat.cols(), n_manual);
        ASSERT_EQ(mat.non_zeros(), nnz_manual);

        ASSERT_EQ(
            mat.get_elem(0, 0).get_scalar(), static_cast<TPrecision>(-5)
        );
        ASSERT_EQ(
            mat.get_elem(1, 0).get_scalar(), static_cast<TPrecision>(3)
        );
        ASSERT_EQ(
            mat.get_elem(0, 1).get_scalar(), static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            mat.get_elem(1, 1).get_scalar(), static_cast<TPrecision>(3.5)
        );
        ASSERT_EQ(
            mat.get_elem(0, 2).get_scalar(), static_cast<TPrecision>(-20)
        );
        ASSERT_EQ(
            mat.get_elem(1, 2).get_scalar(), static_cast<TPrecision>(0)
        );
    
    }

    void TestBadDynamicMemConstruction() {

        int *h_col_offsets_dummy = nullptr;
        int *h_row_indices_dummy = nullptr;
        double *h_vals_dummy = nullptr;

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() {
                NoFillMatrixSparse<double>(
                    TestBase::bundle,
                    h_col_offsets_dummy, h_row_indices_dummy, h_vals_dummy,
                    -1, 4, 4
                );
            }
        );

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() {
                NoFillMatrixSparse<double>(
                    TestBase::bundle,
                    h_col_offsets_dummy, h_row_indices_dummy, h_vals_dummy,
                    4, -2, 4
                );
            }
        );

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() {
                NoFillMatrixSparse<double>(
                    TestBase::bundle,
                    h_col_offsets_dummy, h_row_indices_dummy, h_vals_dummy,
                    4, 4, -1
                );
            }
        );

    }

    template <typename TPrecision>
    void TestDynamicMemCopyToPtr() {
    
        const int m_manual(2);
        const int n_manual(3);
        const int nnz_manual(4);

        NoFillMatrixSparse<TPrecision> mat_manual(
            TestBase::bundle,
            {{static_cast<TPrecision>(-5), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(-20)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(3.5),
              static_cast<TPrecision>(0.)}}
        );

        int *h_col_offsets = static_cast<int *>(
            malloc((n_manual+1)*sizeof(int))
        );
        int *h_row_indices = static_cast<int *>(
            malloc(nnz_manual*sizeof(int))
        );
        TPrecision *h_vals = static_cast<TPrecision *>(
            malloc(nnz_manual*sizeof(TPrecision))
        );

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

        ASSERT_EQ(h_vals[0], static_cast<TPrecision>(-5));
        ASSERT_EQ(h_vals[1], static_cast<TPrecision>(3));
        ASSERT_EQ(h_vals[2], static_cast<TPrecision>(3.5));
        ASSERT_EQ(h_vals[3], static_cast<TPrecision>(-20));

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);
    
        const int m_manual_2(3);
        const int n_manual_2(5);
        const int nnz_manual_2(8);

        NoFillMatrixSparse<TPrecision> mat_manual_2(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(0),
              static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(9),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(14),
              static_cast<TPrecision>(0)}}
        );

        int *h_col_offsets_2 = static_cast<int *>(
            malloc((n_manual_2+1)*sizeof(int))
        );
        int *h_row_indices_2 = static_cast<int *>(
            malloc(nnz_manual_2*sizeof(int))
        );
        TPrecision *h_vals_2 = static_cast<TPrecision *>(
            malloc(nnz_manual_2*sizeof(TPrecision))
        );

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

        ASSERT_EQ(h_vals_2[0], static_cast<TPrecision>(1));
        ASSERT_EQ(h_vals_2[1], static_cast<TPrecision>(6));
        ASSERT_EQ(h_vals_2[2], static_cast<TPrecision>(2));
        ASSERT_EQ(h_vals_2[3], static_cast<TPrecision>(3));
        ASSERT_EQ(h_vals_2[4], static_cast<TPrecision>(9));
        ASSERT_EQ(h_vals_2[5], static_cast<TPrecision>(14));
        ASSERT_EQ(h_vals_2[6], static_cast<TPrecision>(5));
        ASSERT_EQ(h_vals_2[7], static_cast<TPrecision>(10));

        free(h_col_offsets_2);
        free(h_row_indices_2);
        free(h_vals_2);

    }

    void TestBadDynamicMemCopyToPtr() {

        const int m_manual(4);
        const int n_manual(5);
        const int nnz_manual(6);
        NoFillMatrixSparse<double> mat(
            TestBase::bundle,
            {{1., 0., 0., 0., 0.},
             {0., 3.5, 20., 0., 0.},
             {0., 2., 0., 0., 0.},
             {-1., 0., 0., 0., -0.5}}
        );

        int *h_col_offsets = static_cast<int *>(
            malloc((n_manual+1)*sizeof(int))
        );
        int *h_row_indices = static_cast<int *>(
            malloc(nnz_manual*sizeof(int))
        );
        double *h_vals = static_cast<double *>(
            malloc(nnz_manual*sizeof(double))
        );
        
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

    template <typename TPrecision>
    void TestRandomMatrixCreation() {

        constexpr int m_rand(30);
        constexpr int n_rand(40);
        constexpr double miss_tol(0.07);

        // Check that zero fill_prob just gives diagonal matrix
        NoFillMatrixSparse<TPrecision> test_rand_just_diag(
            NoFillMatrixSparse<TPrecision>::Random(
                TestBase::bundle, m_rand, n_rand, 0.
            )
        );
        ASSERT_EQ(test_rand_just_diag.rows(), m_rand);
        ASSERT_EQ(test_rand_just_diag.cols(), n_rand);
        ASSERT_EQ(test_rand_just_diag.non_zeros(), m_rand);

        // Just test for non-zero fill_prob gives right size and numbers aren't
        // generally the same (check middle numbers are different from 5
        // adjacent above and below or all are equally zero) also check if fill
        // probability is right with 5% error
        NoFillMatrixSparse<TPrecision> test_rand_full(
            NoFillMatrixSparse<TPrecision>::Random(
                TestBase::bundle, m_rand, n_rand, 1.
            )
        );
        ASSERT_EQ(test_rand_full.rows(), m_rand);
        ASSERT_EQ(test_rand_full.cols(), n_rand);
        for (int j=1; j<n_rand-1; ++j) {
            for (int i=1; i<m_rand-1; ++i) {
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
                     ((test_rand_full.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }
        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_full.non_zeros())/
             static_cast<TPrecision>(
                test_rand_full.rows() *
                test_rand_full.cols()
             )),
            1.,
            miss_tol
        );

        NoFillMatrixSparse<TPrecision> test_rand_67(
            NoFillMatrixSparse<TPrecision>::Random(
                TestBase::bundle, m_rand, n_rand, 0.67
            )
        );
        ASSERT_EQ(test_rand_67.rows(), m_rand);
        ASSERT_EQ(test_rand_67.cols(), n_rand);
        for (int j=1; j<n_rand-1; ++j) {
            for (int i=1; i<m_rand-1; ++i) {
                ASSERT_TRUE(
                    (((test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_67.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }
        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_67.non_zeros())/
             static_cast<TPrecision>(test_rand_67.rows()*test_rand_67.cols())),
            0.67,
            miss_tol
        );

        NoFillMatrixSparse<TPrecision> test_rand_50(
            NoFillMatrixSparse<TPrecision>::Random(
                TestBase::bundle, m_rand, n_rand, 0.5
            )
        );
        ASSERT_EQ(test_rand_50.rows(), m_rand);
        ASSERT_EQ(test_rand_50.cols(), n_rand);
        for (int j=1; j<n_rand-1; ++j) {
            for (int i=1; i<m_rand-1; ++i) {
                ASSERT_TRUE(
                    (((test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_50.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }
        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_50.non_zeros())/
             static_cast<TPrecision>(test_rand_50.rows()*test_rand_50.cols())),
            0.5,
            miss_tol
        );

    }

    template <typename TPrecision>
    void TestRandomUTMatrixCreation(int m_rand, int n_rand) {

        constexpr double miss_tol(0.1);
        int min_dim;
        int max_non_zeros;
        if (m_rand >= n_rand) {
            min_dim = n_rand;
            max_non_zeros = (n_rand*n_rand-n_rand)/2;
        } else {
            max_non_zeros = (m_rand*m_rand-m_rand)/2+(n_rand-m_rand)*m_rand;
            min_dim = m_rand;
        }
        

        // Check that zero fill_prob just gives diagonal matrix
        NoFillMatrixSparse<TPrecision> test_rand_just_diag(
            NoFillMatrixSparse<TPrecision>::Random_UT(
                TestBase::bundle, m_rand, n_rand, 0.
            )
        );
        ASSERT_EQ(test_rand_just_diag.rows(), m_rand);
        ASSERT_EQ(test_rand_just_diag.cols(), n_rand);
        ASSERT_EQ(test_rand_just_diag.non_zeros(), min_dim);

        // Just test for non-zero fill_prob gives right size and numbers aren't
        // generally the same (check middle numbers are different from 5
        // adjacent above and below or all are equally zero) also check if fill
        // probability is right with 5% error
        NoFillMatrixSparse<TPrecision> test_rand_full(
            NoFillMatrixSparse<TPrecision>::Random_UT(
                TestBase::bundle, m_rand, n_rand, 1.
            )
        );

        ASSERT_EQ(test_rand_full.rows(), m_rand);
        ASSERT_EQ(test_rand_full.cols(), n_rand);

        for (int j=1; j<n_rand-1; ++j) {
            for (int i=1; ((i < j) && (i < m_rand-1)); ++i) {
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
                     ((test_rand_full.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }

        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_full.non_zeros()-m_rand) /
             static_cast<TPrecision>(max_non_zeros)),
            1.,
            miss_tol
        );

        // Check non-zero diagonal
        for (int i=0; i<min_dim; ++i) {
            ASSERT_FALSE(
                test_rand_full.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero below diagonal
        for (int j=0; j<n_rand; ++j) {
            for (int i=j+1; i<m_rand; ++i) {
                ASSERT_EQ(
                    test_rand_full.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

        NoFillMatrixSparse<TPrecision> test_rand_67(
            NoFillMatrixSparse<TPrecision>::Random_UT(
                TestBase::bundle, m_rand, n_rand, 0.67
            )
        );

        ASSERT_EQ(test_rand_67.rows(), m_rand);
        ASSERT_EQ(test_rand_67.cols(), n_rand);

        for (int j=1; j<n_rand-1; ++j) {
            for (int i=1; ((i < j) && (i < m_rand-1)); ++i) {
                ASSERT_TRUE(
                    (((test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_67.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }

        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_67.non_zeros()-m_rand) /
             static_cast<TPrecision>(max_non_zeros)),
            0.67,
            miss_tol
        );

        // Check non-zero diagonal
        for (int i=0; i<min_dim; ++i) {
            ASSERT_FALSE(
                test_rand_67.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero below diagonal
        for (int j=0; j<n_rand; ++j) {
            for (int i=j+1; i<m_rand; ++i) {
                ASSERT_EQ(
                    test_rand_67.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

        NoFillMatrixSparse<TPrecision> test_rand_50(
            NoFillMatrixSparse<TPrecision>::Random_UT(
                TestBase::bundle, m_rand, n_rand, 0.5
            )
        );

        ASSERT_EQ(test_rand_50.rows(), m_rand);
        ASSERT_EQ(test_rand_50.cols(), n_rand);

        for (int j=1; j<n_rand-1; ++j) {
            for (int i=1; ((i < j) && (i < m_rand-1)); ++i) {
                ASSERT_TRUE(
                    (((test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_50.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }

        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_50.non_zeros()-min_dim) /
             static_cast<TPrecision>(static_cast<TPrecision>(max_non_zeros))),
            0.5,
            miss_tol
        );

        // Check non-zero diagonal
        for (int i=0; i<min_dim; ++i) {
            ASSERT_FALSE(
                test_rand_50.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero below diagonal
        for (int j=0; j<n_rand; ++j) {
            for (int i=j+1; i<m_rand; ++i) {
                ASSERT_EQ(
                    test_rand_67.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

    }

    template <typename TPrecision>
    void TestRandomLTMatrixCreation(int m_rand, int n_rand) {

        constexpr double miss_tol(0.1);
        int min_dim;
        int max_non_zeros;
        if (m_rand >= n_rand) {
            max_non_zeros = (n_rand*n_rand-n_rand)/2+(m_rand-n_rand)*n_rand;
            min_dim = n_rand;
        } else {
            max_non_zeros = (m_rand*m_rand-m_rand)/2;
            min_dim = m_rand;
        }

        // Check that zero fill_prob just gives diagonal matrix
        NoFillMatrixSparse<TPrecision> test_rand_just_diag(
            NoFillMatrixSparse<TPrecision>::Random_LT(
                TestBase::bundle, m_rand, n_rand, 0.
            )
        );
        ASSERT_EQ(test_rand_just_diag.rows(), m_rand);
        ASSERT_EQ(test_rand_just_diag.cols(), n_rand);
        ASSERT_EQ(test_rand_just_diag.non_zeros(), min_dim);

        // Just test for non-zero fill_prob gives right size and numbers aren't
        // generally the same (check middle numbers are different from 5
        // adjacent above and below or all are equally zero) also check if fill
        // probability is right with 5% error
        NoFillMatrixSparse<TPrecision> test_rand_full(
            NoFillMatrixSparse<TPrecision>::Random_LT(
                TestBase::bundle, m_rand, n_rand, 1.
            )
        );

        ASSERT_EQ(test_rand_full.rows(), m_rand);
        ASSERT_EQ(test_rand_full.cols(), n_rand);

        for (int j=1; j<n_rand-1; ++j) {
            for (int i=j+1; i<m_rand-1; ++i) {
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
                     ((test_rand_full.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_full.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }

        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_full.non_zeros()-min_dim) /
             static_cast<TPrecision>(max_non_zeros)),
            1.,
            miss_tol
        );

        // Check non-zero diagonal
        for (int i=0; i<min_dim; ++i) {
            ASSERT_FALSE(
                test_rand_full.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero above diagonal
        for (int j=0; j<n_rand; ++j) {
            for (int i=0; ((i < j) && (i < m_rand)); ++i) {
                ASSERT_EQ(
                    test_rand_full.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

        NoFillMatrixSparse<TPrecision> test_rand_67(
            NoFillMatrixSparse<TPrecision>::Random_LT(
                TestBase::bundle, m_rand, n_rand, 0.67
        
         )
        );

        ASSERT_EQ(test_rand_67.rows(), m_rand);
        ASSERT_EQ(test_rand_67.cols(), n_rand);

        for (int j=1; j<n_rand-1; ++j) {
            for (int i=j+1; i<m_rand-1; ++i) {
                ASSERT_TRUE(
                    (((test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_67.get_elem(i, j).get_scalar() !=
                       test_rand_67.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_67.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_67.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }

        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_67.non_zeros()-min_dim) /
             static_cast<TPrecision>(max_non_zeros)),
            0.67,
            miss_tol
        );

        // Check non-zero diagonal
        for (int i=0; i<min_dim; ++i) {
            ASSERT_FALSE(
                test_rand_67.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero above diagonal
        for (int j=0; j<n_rand; ++j) {
            for (int i=0; ((i < j) && (i < m_rand)); ++i) {
                ASSERT_EQ(
                    test_rand_67.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

        NoFillMatrixSparse<TPrecision> test_rand_50(
            NoFillMatrixSparse<TPrecision>::Random_LT(
                TestBase::bundle, m_rand, n_rand, 0.5
            )
        );

        ASSERT_EQ(test_rand_50.rows(), m_rand);
        ASSERT_EQ(test_rand_50.cols(), n_rand);

        for (int j=1; j<n_rand-1; ++j) {
            for (int i=j+1; i<m_rand-1; ++i) {
                ASSERT_TRUE(
                    (((test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i-1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i+1, j).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i, j-1).get_scalar()) ||
                      (test_rand_50.get_elem(i, j).get_scalar() !=
                       test_rand_50.get_elem(i, j+1).get_scalar()))
                     ||
                     ((test_rand_50.get_elem(i, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i-1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i+1, j).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i, j-1).get_scalar() ==
                       static_cast<TPrecision>(0.)) &&
                      (test_rand_50.get_elem(i, j+1).get_scalar() ==
                       static_cast<TPrecision>(0.))))
                );
            }
        }

        ASSERT_NEAR(
            (static_cast<TPrecision>(test_rand_50.non_zeros()-min_dim) /
             static_cast<TPrecision>(static_cast<TPrecision>(max_non_zeros))),
            0.5,
            miss_tol
        );

        // Check non-zero diagonal
        for (int i=0; i<min_dim; ++i) {
            ASSERT_FALSE(
                test_rand_50.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero above diagonal
        for (int j=0; j<n_rand; ++j) {
            for (int i=0; ((i < j) && (i < m_rand)); ++i) {
                ASSERT_EQ(
                    test_rand_67.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

    }

    template <typename TPrecision>
    void TestBlock() {

        const NoFillMatrixSparse<TPrecision> mat (
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(8), static_cast<TPrecision>(9),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(19),
              static_cast<TPrecision>(20)}}
        );

        // Test copy constructor and access for block 0, 0, 4, 2
        typename NoFillMatrixSparse<TPrecision>::Block blk_0_0_4_2(
            mat.get_block(0, 0, 4, 2)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(1)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(2, 0).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(3, 0).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(2)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(2, 1).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(3, 1).get_scalar(),
            static_cast<TPrecision>(0)
        );

        // Test copy constructor and access for block 2, 1, 2, 3
        typename NoFillMatrixSparse<TPrecision>::Block blk_2_1_2_3(
            mat.get_block(2, 1, 2, 3)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(0, 2).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(0)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(1, 2).get_scalar(),
            static_cast<TPrecision>(19)
        );
    
    }

    // template <typename TPrecision>
    // void TestRandomMatVec() {

    //     // Test random
    //     const int m_rand(3);
    //     const int n_rand(4);
    //     const double fill_ratio_rand(0.5);
    //     NoFillMatrixSparse<TPrecision> rand_mat(
    //         NoFillMatrixSparse<TPrecision>::Random(
    //             TestBase::bundle, m_rand, n_rand, fill_ratio_rand
    //         )
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat*Vector<TPrecision>(
    //             TestBase::bundle,
    //             {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
    //              static_cast<TPrecision>(0), static_cast<TPrecision>(0)}
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {rand_mat.get_elem(0, 0).get_scalar(),
    //              rand_mat.get_elem(1, 0).get_scalar(),
    //              rand_mat.get_elem(2, 0).get_scalar()}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat*Vector<TPrecision>(
    //             TestBase::bundle,
    //             {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
    //              static_cast<TPrecision>(0), static_cast<TPrecision>(0)}
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {rand_mat.get_elem(0, 1).get_scalar(),
    //              rand_mat.get_elem(1, 1).get_scalar(),
    //              rand_mat.get_elem(2, 1).get_scalar()}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat*Vector<TPrecision>(
    //             TestBase::bundle,
    //             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
    //              static_cast<TPrecision>(1), static_cast<TPrecision>(0)}
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {rand_mat.get_elem(0, 2).get_scalar(),
    //              rand_mat.get_elem(1, 2).get_scalar(),
    //              rand_mat.get_elem(2, 2).get_scalar()}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat*Vector<TPrecision>(
    //             TestBase::bundle,
    //             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
    //              static_cast<TPrecision>(0), static_cast<TPrecision>(1)}
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {rand_mat.get_elem(0, 3).get_scalar(),
    //              rand_mat.get_elem(1, 3).get_scalar(),
    //              rand_mat.get_elem(2, 3).get_scalar()}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat*Vector<TPrecision>(
    //             TestBase::bundle,
    //             {static_cast<TPrecision>(1), static_cast<TPrecision>(0.1),
    //              static_cast<TPrecision>(0.01), static_cast<TPrecision>(0.001)}
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {((static_cast<TPrecision>(1) *
    //                rand_mat.get_elem(0, 0).get_scalar()) +
    //               (static_cast<TPrecision>(0.1) *
    //                rand_mat.get_elem(0, 1).get_scalar()) +
    //               (static_cast<TPrecision>(0.01) *
    //                rand_mat.get_elem(0, 2).get_scalar()) +
    //               (static_cast<TPrecision>(0.001) *
    //                rand_mat.get_elem(0, 3).get_scalar())),
    //              ((static_cast<TPrecision>(1) *
    //                rand_mat.get_elem(1, 0).get_scalar()) +
    //               (static_cast<TPrecision>(0.1) *
    //                rand_mat.get_elem(1, 1).get_scalar()) +
    //               (static_cast<TPrecision>(0.01) *
    //                rand_mat.get_elem(1, 2).get_scalar())+
    //               (static_cast<TPrecision>(0.001) *
    //                rand_mat.get_elem(1, 3).get_scalar())),
    //              ((static_cast<TPrecision>(1) *
    //                rand_mat.get_elem(2, 0).get_scalar()) +
    //               (static_cast<TPrecision>(0.1) *
    //                rand_mat.get_elem(2, 1).get_scalar()) +
    //               (static_cast<TPrecision>(0.01) *
    //                rand_mat.get_elem(2, 2).get_scalar())+
    //               (static_cast<TPrecision>(0.001) *
    //                rand_mat.get_elem(2, 3).get_scalar()))}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );

    // }

    // template <typename TPrecision>
    // void TestRandomTransposeMatVec() {

    //     // Test random
    //     const int m_rand(3);
    //     const int n_rand(2);
    //     const double fill_ratio_rand(0.5);
    //     NoFillMatrixSparse<TPrecision> rand_mat(
    //         NoFillMatrixSparse<TPrecision>::Random(
    //             TestBase::bundle, m_rand, n_rand, fill_ratio_rand
    //         )
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat.transpose_prod(
    //             Vector<TPrecision>(
    //                 TestBase::bundle,
    //                 {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
    //                  static_cast<TPrecision>(0)}
    //             )
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {rand_mat.get_elem(0, 0).get_scalar(),
    //              rand_mat.get_elem(0, 1).get_scalar()}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat.transpose_prod(
    //             Vector<TPrecision>(
    //                 TestBase::bundle,
    //                 {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
    //                  static_cast<TPrecision>(0)}
    //             )
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {rand_mat.get_elem(1, 0).get_scalar(),
    //              rand_mat.get_elem(1, 1).get_scalar()}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat.transpose_prod(
    //             Vector<TPrecision>(
    //                 TestBase::bundle,
    //                 {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
    //                  static_cast<TPrecision>(1)}
    //             )
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {rand_mat.get_elem(2, 0).get_scalar(),
    //              rand_mat.get_elem(2, 1).get_scalar()}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );
    //     ASSERT_VECTOR_NEAR(
    //         rand_mat.transpose_prod(
    //             Vector<TPrecision>(
    //                 TestBase::bundle,
    //                 {static_cast<TPrecision>(1), static_cast<TPrecision>(0.1),
    //                  static_cast<TPrecision>(0.01)}
    //             )
    //         ),
    //         Vector<TPrecision>(
    //             TestBase::bundle,
    //             {((static_cast<TPrecision>(1) *
    //                rand_mat.get_elem(0, 0).get_scalar()) +
    //               (static_cast<TPrecision>(0.1) *
    //                rand_mat.get_elem(1, 0).get_scalar()) +
    //               (static_cast<TPrecision>(0.01) *
    //                rand_mat.get_elem(2, 0).get_scalar())),
    //              ((static_cast<TPrecision>(1) *
    //                rand_mat.get_elem(0, 1).get_scalar()) +
    //               (static_cast<TPrecision>(0.1) *
    //                rand_mat.get_elem(1, 1).get_scalar()) +
    //               (static_cast<TPrecision>(0.01) *
    //                rand_mat.get_elem(2, 1).get_scalar()))}
    //         ),
    //         (static_cast<TPrecision>(2.) *
    //          static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
    //     );

    // }

    template <typename TPrecision>
    void TestRandomTranspose() {

        constexpr int m_rand(4);
        constexpr int n_rand(3);
        NoFillMatrixSparse<TPrecision> mat(
            NoFillMatrixSparse<TPrecision>::Random(
                TestBase::bundle, n_rand, m_rand, 0.67
            )
        );

        NoFillMatrixSparse<TPrecision> mat_transposed(mat.transpose());
        ASSERT_EQ(mat_transposed.rows(), m_rand);
        ASSERT_EQ(mat_transposed.cols(), n_rand);
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(
                    mat_transposed.get_elem(i, j).get_scalar(),
                    mat.get_elem(j, i).get_scalar()
                );
            }
        }

    }

};

TEST_F(NoFillMatrixSparse_Test, TestPropertyAccess) {
    TestPropertyAccess<__half>();
    TestPropertyAccess<float>();
    TestPropertyAccess<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestNonZeros) {
    TestNonZeros<__half>();
    TestNonZeros<float>();
    TestNonZeros<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestPrintAndInfoString) {
    TestPrintAndInfoString();
}

TEST_F(NoFillMatrixSparse_Test, TestConstruction) {
    TestConstruction<__half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestBadConstruction) {
    TestBadConstruction();
}

TEST_F(NoFillMatrixSparse_Test, TestListInitialization) {
    TestListInitialization<__half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestBadListInitialization) {
    TestBadListInitialization();
}

TEST_F(NoFillMatrixSparse_Test, TestCoeffJustGetAccess) {
    TestCoeffJustGetAccess<__half>();
    TestCoeffJustGetAccess<float>();
    TestCoeffJustGetAccess<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestBadCoeffJustGetAccess) {
    TestBadCoeffJustGetAccess();
}

TEST_F(NoFillMatrixSparse_Test, TestCopyAssignment) {
    TestCopyAssignment<__half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestCopyConstructor) {
    TestCopyConstructor<__half>();
    TestCopyConstructor<float>();
    TestCopyConstructor<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestDynamicMemConstruction) {
    TestDynamicMemConstruction<__half>();
    TestDynamicMemConstruction<float>();
    TestDynamicMemConstruction<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestBadDynamicMemConstruction) {
    TestBadDynamicMemConstruction();
}

TEST_F(NoFillMatrixSparse_Test, TestDynamicMemCopyToPtr) {
    TestDynamicMemCopyToPtr<__half>();
    TestDynamicMemCopyToPtr<float>();
    TestDynamicMemCopyToPtr<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestBadDynamicMemCopyToPtr) {
    TestBadDynamicMemCopyToPtr();
}

TEST_F(NoFillMatrixSparse_Test, TestZeroMatrixCreation) {
    TestZeroMatrixCreation<__half>();
    TestZeroMatrixCreation<float>();
    TestZeroMatrixCreation<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestOnesMatrixCreation) {
    TestOnesMatrixCreation<__half>();
    TestOnesMatrixCreation<float>();
    TestOnesMatrixCreation<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestIdentityMatrixCreation) {
    TestIdentityMatrixCreation<__half>();
    TestIdentityMatrixCreation<float>();
    TestIdentityMatrixCreation<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestRandomMatrixCreation) {
    TestRandomMatrixCreation<__half>();
    TestRandomMatrixCreation<float>();
    TestRandomMatrixCreation<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestRandomUTMatrixCreation) {
    TestRandomUTMatrixCreation<__half>(30, 40);
    TestRandomUTMatrixCreation<__half>(40, 30);
    TestRandomUTMatrixCreation<float>(30, 40);
    TestRandomUTMatrixCreation<float>(40, 30);
    TestRandomUTMatrixCreation<double>(30, 40);
    TestRandomUTMatrixCreation<double>(40, 30);
}

TEST_F(NoFillMatrixSparse_Test, TestRandomLTMatrixCreation) {
    TestRandomLTMatrixCreation<__half>(30, 40);
    TestRandomLTMatrixCreation<__half>(40, 30);
    TestRandomLTMatrixCreation<float>(30, 40);
    TestRandomLTMatrixCreation<float>(40, 30);
    TestRandomLTMatrixCreation<double>(30, 40);
    TestRandomLTMatrixCreation<double>(40, 30);
}

TEST_F(NoFillMatrixSparse_Test, TestCol) {
    TestCol<__half>();
    TestCol<float>();
    TestCol<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestBadCol) {
    TestBadCol();
}

TEST_F(NoFillMatrixSparse_Test, TestBlock) {
    TestBlock<__half>();
    TestBlock<float>();
    TestBlock<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestBadBlock) {
    TestBadBlock();
}

// TEST_F(NoFillMatrixSparse_Test, TestScale) {
//     TestScale<__half>();
//     TestScale<float>();
//     TestScale<double>();
// }

// TEST_F(NoFillMatrixSparse_Test, TestScaleAssignment) {
//     TestScaleAssignment<__half>();
//     TestScaleAssignment<float>();
//     TestScaleAssignment<double>();
// }

TEST_F(NoFillMatrixSparse_Test, TestMaxMagElem) {
    TestMaxMagElem<__half>();
    TestMaxMagElem<float>();
    TestMaxMagElem<double>();
}

// TEST_F(NoFillMatrixSparse_Test, TestNormalizeMagnitude) {
//     TestNormalizeMagnitude<__half>();
//     TestNormalizeMagnitude<float>();
//     TestNormalizeMagnitude<double>();
// }

// TEST_F(NoFillMatrixSparse_Test, TestMatVec) {
//     TestMatVec<__half>();
//     TestMatVec<float>();
//     TestMatVec<double>();
// }

// TEST_F(NoFillMatrixSparse_Test, TestRandomMatVec) {
//     TestRandomMatVec<__half>();
//     TestRandomMatVec<float>();
//     TestRandomMatVec<double>();
// }

// TEST_F(NoFillMatrixSparse_Test, TestBadMatVec) {
//     TestBadMatVec<__half>();
//     TestBadMatVec<float>();
//     TestBadMatVec<double>();
// }

// TEST_F(NoFillMatrixSparse_Test, TestTransposeMatVec) {
//     TestTransposeMatVec<__half>();
//     TestTransposeMatVec<float>();
//     TestTransposeMatVec<double>();
// }

// TEST_F(NoFillMatrixSparse_Test, TestRandomTransposeMatVec) {
//     TestRandomTransposeMatVec<__half>();
//     TestRandomTransposeMatVec<float>();
//     TestRandomTransposeMatVec<double>();
// }

// TEST_F(NoFillMatrixSparse_Test, TestBadTransposeMatVec) {
//     TestBadTransposeMatVec<__half>();
//     TestBadTransposeMatVec<float>();
//     TestBadTransposeMatVec<double>();
// }

TEST_F(NoFillMatrixSparse_Test, TestTranspose) {
    TestTranspose<__half>();
    TestTranspose<float>();
    TestTranspose<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestRandomTranspose) {
    TestRandomTranspose<__half>();
    TestRandomTranspose<float>();
    TestRandomTranspose<double>();
}

TEST_F(NoFillMatrixSparse_Test, TestCast) {
    TestCast();
}

// class NoFillMatrixSparse_Substitution_Test:
//     public Matrix_Substitution_Test<NoFillMatrixSparse>
// {};

// TEST_F(NoFillMatrixSparse_Substitution_Test, TestForwardSubstitution) {
//     TestForwardSubstitution<__half>();
//     TestForwardSubstitution<float>();
//     TestForwardSubstitution<double>();
// }

// TEST_F(NoFillMatrixSparse_Substitution_Test, TestRandomForwardSubstitution) {
//     TestRandomForwardSubstitution<__half>();
//     TestRandomForwardSubstitution<float>();
//     TestRandomForwardSubstitution<double>();
// }

// TEST_F(NoFillMatrixSparse_Substitution_Test, TestRandomSparseForwardSubstitution) {
//     TestRandomSparseForwardSubstitution<__half>();
//     TestRandomSparseForwardSubstitution<float>();
//     TestRandomSparseForwardSubstitution<double>();
// }

// TEST_F(NoFillMatrixSparse_Substitution_Test, TestBackwardSubstitution) {
//     TestBackwardSubstitution<__half>();
//     TestBackwardSubstitution<float>();
//     TestBackwardSubstitution<double>();
// }

// TEST_F(NoFillMatrixSparse_Substitution_Test, TestRandomBackwardSubstitution) {
//     TestRandomBackwardSubstitution<__half>();
//     TestRandomBackwardSubstitution<float>();
//     TestRandomBackwardSubstitution<double>();
// }

// TEST_F(NoFillMatrixSparse_Substitution_Test, TestRandomSparseBackwardSubstitution) {
//     TestRandomSparseBackwardSubstitution<__half>();
//     TestRandomSparseBackwardSubstitution<float>();
//     TestRandomSparseBackwardSubstitution<double>();
// }