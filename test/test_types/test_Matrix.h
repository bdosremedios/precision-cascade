#include "../test.h"

template <template <typename> typename M>
class Matrix_Test: public TestBase
{
protected:

    template <typename T>
    void TestCoeffAccess() {
        
        constexpr int m(24);
        constexpr int n(12);
        M<T> test_mat(TestBase::bundle, m, n);

        Scalar<T> elem(static_cast<T>(1));
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                test_mat.set_elem(i, j, elem);
                elem += Scalar<T>(static_cast<T>(1));
            }
        }

        T test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), test_elem);
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 0 row as all -1
        for (int j=0; j<n; ++j) {
            test_mat.set_elem(0, j, Scalar<T>(static_cast<T>(-1.)));
        }

        // Test matches modified matrix
        test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 4 row as decreasing by -1 from -1
        Scalar<T> row_5_elem(static_cast<T>(-1.));
        for (int j=0; j<n; ++j) {
            test_mat.set_elem(4, j, row_5_elem);
            row_5_elem += Scalar<T>(static_cast<T>(-1.));
        }

        // Test matches modified matrix
        test_elem = static_cast<T>(1);
        T row_5_test_elem = static_cast<T>(-1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else if (i == 4) {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), row_5_test_elem);
                    row_5_test_elem += static_cast<T>(-1.);
                } else {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 2 col as incresing by 1 from -5
        Scalar<T> coL_3_elem(static_cast<T>(-5.));
        for (int i=0; i<m; ++i) {
            test_mat.set_elem(i, 2, coL_3_elem);
            coL_3_elem += Scalar<T>(static_cast<T>(1.));

        }

    
    // Test matches modified matrix
        test_elem = static_cast<T>(1);
        row_5_test_elem = static_cast<T>(-1.);
        T coL_3_test_elem = static_cast<T>(-5.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (j == 2) {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), coL_3_test_elem);
                    coL_3_test_elem += static_cast<T>(1.);
                } else if (i == 0) {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else if (i == 4) {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), row_5_test_elem);
                    row_5_test_elem += static_cast<T>(-1.);
                    if (j == 1) { row_5_test_elem += static_cast<T>(-1.); }
                } else {
                    ASSERT_EQ(test_mat.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

    }

    void TestBadCoeffAccess() {
        
        constexpr int m(24);
        constexpr int n(12);
        M<double> test_mat(M<double>::Random(TestBase::bundle, m, n));

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(0, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(0, n); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(-1, 0); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(m, 0); });

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(0, -1, Scalar<double>(0.)); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(0, n, Scalar<double>(0.)); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(-1, 0, Scalar<double>(0.)); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(m, 0, Scalar<double>(0.)); }
        );

    }

    template <typename T>
    void TestPropertyAccess() {

        constexpr int m_1(62);
        constexpr int n_1(3);
        M<T> mat_1(TestBase::bundle, m_1, n_1);
        ASSERT_EQ(mat_1.rows(), m_1);
        ASSERT_EQ(mat_1.cols(), n_1);

        constexpr int m_2(15);
        constexpr int n_2(90);
        M<T> mat_2(TestBase::bundle, m_2, n_2);
        ASSERT_EQ(mat_2.rows(), m_2);
        ASSERT_EQ(mat_2.cols(), n_2);

        constexpr int m_3(20);
        constexpr int n_3(20);
        M<T> mat_3(TestBase::bundle, m_3, n_3);
        ASSERT_EQ(mat_3.rows(), m_3);
        ASSERT_EQ(mat_3.cols(), n_3);

    }

    template <typename T>
    void TestConstruction() {

        M<T> test_mat_empty(TestBase::bundle);
        ASSERT_EQ(test_mat_empty.rows(), 0);
        ASSERT_EQ(test_mat_empty.cols(), 0);

        constexpr int m(12);
        M<T> test_mat_square(TestBase::bundle, m, m);
        ASSERT_EQ(test_mat_square.rows(), m);
        ASSERT_EQ(test_mat_square.cols(), m);

        constexpr int n(33);
        M<T> test_mat_wide(TestBase::bundle, m, n);
        ASSERT_EQ(test_mat_wide.rows(), m);
        ASSERT_EQ(test_mat_wide.cols(), n);

        M<T> test_mat_tall(TestBase::bundle, n, m);
        ASSERT_EQ(test_mat_tall.rows(), n);
        ASSERT_EQ(test_mat_tall.cols(), m);

    }

    template <typename T>
    void TestListInitialization() {

        M<T> test_mat_0_0 (TestBase::bundle, {});
        ASSERT_EQ(test_mat_0_0.rows(), 0);
        ASSERT_EQ(test_mat_0_0.cols(), 0);

        M<T> test_mat_0_1 (TestBase::bundle, {{}});
        ASSERT_EQ(test_mat_0_1.rows(), 1);
        ASSERT_EQ(test_mat_0_1.cols(), 0);

        M<T> test_mat_1(
            TestBase::bundle,
            {{static_cast<T>(5.), static_cast<T>(3.), static_cast<T>(27.)},
             {static_cast<T>(88.), static_cast<T>(-4.), static_cast<T>(-6.)},
             {static_cast<T>(100.), static_cast<T>(12.), static_cast<T>(2.)}}
        );
        ASSERT_EQ(test_mat_1.rows(), 3);
        ASSERT_EQ(test_mat_1.cols(), 3);
        ASSERT_EQ(test_mat_1.get_elem(0, 0).get_scalar(), static_cast<T>(5.));
        ASSERT_EQ(test_mat_1.get_elem(0, 1).get_scalar(), static_cast<T>(3.));
        ASSERT_EQ(test_mat_1.get_elem(0, 2).get_scalar(), static_cast<T>(27.));
        ASSERT_EQ(test_mat_1.get_elem(1, 0).get_scalar(), static_cast<T>(88.));
        ASSERT_EQ(test_mat_1.get_elem(1, 1).get_scalar(), static_cast<T>(-4.));
        ASSERT_EQ(test_mat_1.get_elem(1, 2).get_scalar(), static_cast<T>(-6.));
        ASSERT_EQ(test_mat_1.get_elem(2, 0).get_scalar(), static_cast<T>(100.));
        ASSERT_EQ(test_mat_1.get_elem(2, 1).get_scalar(), static_cast<T>(12.));
        ASSERT_EQ(test_mat_1.get_elem(2, 2).get_scalar(), static_cast<T>(2.));

        M<T> test_mat_wide(
            TestBase::bundle,
            {{static_cast<T>(7.), static_cast<T>(5.), static_cast<T>(3.)},
             {static_cast<T>(1.), static_cast<T>(6.), static_cast<T>(2.)}}
        );
        ASSERT_EQ(test_mat_wide.rows(), 2);
        ASSERT_EQ(test_mat_wide.cols(), 3);
        ASSERT_EQ(test_mat_wide.get_elem(0, 0).get_scalar(), static_cast<T>(7.));
        ASSERT_EQ(test_mat_wide.get_elem(0, 1).get_scalar(), static_cast<T>(5.));
        ASSERT_EQ(test_mat_wide.get_elem(0, 2).get_scalar(), static_cast<T>(3.));
        ASSERT_EQ(test_mat_wide.get_elem(1, 0).get_scalar(), static_cast<T>(1.));
        ASSERT_EQ(test_mat_wide.get_elem(1, 1).get_scalar(), static_cast<T>(6.));
        ASSERT_EQ(test_mat_wide.get_elem(1, 2).get_scalar(), static_cast<T>(2.));
        
        M<T> test_mat_tall(
            TestBase::bundle,
            {{static_cast<T>(7.), static_cast<T>(5.)},
             {static_cast<T>(1.), static_cast<T>(6.)},
             {static_cast<T>(3.), static_cast<T>(2.)},
             {static_cast<T>(43.), static_cast<T>(9.)}}
        );
        ASSERT_EQ(test_mat_tall.rows(), 4);
        ASSERT_EQ(test_mat_tall.cols(), 2);
        ASSERT_EQ(test_mat_tall.get_elem(0, 0).get_scalar(), static_cast<T>(7.));
        ASSERT_EQ(test_mat_tall.get_elem(0, 1).get_scalar(), static_cast<T>(5.));
        ASSERT_EQ(test_mat_tall.get_elem(1, 0).get_scalar(), static_cast<T>(1.));
        ASSERT_EQ(test_mat_tall.get_elem(1, 1).get_scalar(), static_cast<T>(6.));
        ASSERT_EQ(test_mat_tall.get_elem(2, 0).get_scalar(), static_cast<T>(3.));
        ASSERT_EQ(test_mat_tall.get_elem(2, 1).get_scalar(), static_cast<T>(2.));
        ASSERT_EQ(test_mat_tall.get_elem(3, 0).get_scalar(), static_cast<T>(43.));
        ASSERT_EQ(test_mat_tall.get_elem(3, 1).get_scalar(), static_cast<T>(9.));

    }

    void TestBadListInitialization() {

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(TestBase::bundle, {{1, 2, 3, 4}, {1, 2, 3}}); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(TestBase::bundle, {{1, 2}, {1, 2, 3}}); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(TestBase::bundle, {{1, 2}, {}}); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(TestBase::bundle, {{}, {1, 2, 3}}); }
        );

    }

    template <typename T>
    void TestNonZeros() {

        M<T> test_non_zeros_some(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(0), static_cast<T>(3), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(7), static_cast<T>(0), static_cast<T>(9)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14)},
             {static_cast<T>(16), static_cast<T>(0), static_cast<T>(18), static_cast<T>(0)}}
        );
        ASSERT_EQ(test_non_zeros_some.non_zeros(), 10);

        M<T> test_non_zeros_all(
            TestBase::bundle,
            {{static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)}}
        );
        ASSERT_EQ(test_non_zeros_all.non_zeros(), 0);

        M<T> test_non_zeros_none(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(10)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14), static_cast<T>(15)}}
        );
        ASSERT_EQ(test_non_zeros_none.non_zeros(), 15);

    }

    template <typename T>
    void TestDynamicMemConstruction() {
    
        const int m_manual(2);
        const int n_manual(3);
        T *h_mat_manual = static_cast<T *>(malloc(m_manual*n_manual*sizeof(T)));
        h_mat_manual[0+0*m_manual] = static_cast<T>(-5);
        h_mat_manual[1+0*m_manual] = static_cast<T>(3);
        h_mat_manual[0+1*m_manual] = static_cast<T>(100);
        h_mat_manual[1+1*m_manual] = static_cast<T>(3.5);
        h_mat_manual[0+2*m_manual] = static_cast<T>(-20);
        h_mat_manual[1+2*m_manual] = static_cast<T>(3);

        M<T> test_mat_manual(TestBase::bundle, h_mat_manual, m_manual, n_manual);

        M<T> target_mat_manual(
            TestBase::bundle,
            {{static_cast<T>(-5), static_cast<T>(100), static_cast<T>(-20)},
             {static_cast<T>(3), static_cast<T>(3.5), static_cast<T>(3)}}
        );

        ASSERT_MATRIX_EQ(test_mat_manual, target_mat_manual);

        free(h_mat_manual);
    
        const int m_rand(4);
        const int n_rand(5);
        T *h_mat_rand = static_cast<T *>(malloc(m_rand*n_rand*sizeof(T)));
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                h_mat_rand[i+j*m_rand] = static_cast<T>(rand());
            }
        }

        M<T> test_mat_rand(TestBase::bundle, h_mat_rand, m_rand, n_rand);

        ASSERT_EQ(test_mat_rand.rows(), m_rand);
        ASSERT_EQ(test_mat_rand.cols(), n_rand);
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(test_mat_rand.get_elem(i, j).get_scalar(), h_mat_rand[i+j*m_rand]);
            }
        }

        free(h_mat_rand);

    }

    template <typename T>
    void TestDynamicMemCopyToPtr() {
    
        const int m_manual(2);
        const int n_manual(3);

        M<T> mat_manual(
            TestBase::bundle,
            {{static_cast<T>(-5), static_cast<T>(100), static_cast<T>(-20)},
             {static_cast<T>(3), static_cast<T>(3.5), static_cast<T>(3)}}
        );

        T *h_mat_manual = static_cast<T *>(malloc(m_manual*n_manual*sizeof(T)));
        mat_manual.copy_data_to_ptr(h_mat_manual, m_manual, n_manual);

        ASSERT_EQ(h_mat_manual[0+0*m_manual], static_cast<T>(-5));
        ASSERT_EQ(h_mat_manual[1+0*m_manual], static_cast<T>(3));
        ASSERT_EQ(h_mat_manual[0+1*m_manual], static_cast<T>(100));
        ASSERT_EQ(h_mat_manual[1+1*m_manual], static_cast<T>(3.5));
        ASSERT_EQ(h_mat_manual[0+2*m_manual], static_cast<T>(-20));
        ASSERT_EQ(h_mat_manual[1+2*m_manual], static_cast<T>(3));

        free(h_mat_manual);
    
        const int m_rand(4);
        const int n_rand(5);

        M<T> mat_rand(
            TestBase::bundle,
            {{static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())},
             {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())},
             {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())},
             {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())}}
        );

        T *h_mat_rand = static_cast<T *>(malloc(m_rand*n_rand*sizeof(T)));
        mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand);

        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(h_mat_rand[i+j*m_rand], mat_rand.get_elem(i, j).get_scalar());
            }
        }

        free(h_mat_rand);

    }

    void TestBadDynamicMemCopyToPtr() {

        const int m_rand(4);
        const int n_rand(5);
        M<double> mat_rand(TestBase::bundle, m_rand, n_rand);
        double *h_mat_rand = static_cast<double *>(malloc(m_rand*n_rand*sizeof(double)));
        
        auto try_row_too_small = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand-2, n_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_small);

        auto try_row_too_large = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand+2, n_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_large);

        auto try_col_too_small = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand-2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_col_too_small);

        auto try_col_too_large = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand+2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_col_too_large);

        auto try_match_wrong_dim_row = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, n_rand, n_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_match_wrong_dim_row);

        auto try_match_wrong_dim_col = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, m_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_match_wrong_dim_col);

    }

    template <typename T>
    void TestCopyAssignment() {

        M<T> test_mat_empty(TestBase::bundle, {{}});
        M<T> test_mat_2_2(
            TestBase::bundle,
            {{static_cast<T>(-3.), static_cast<T>(0.)},
             {static_cast<T>(1.), static_cast<T>(10.)}}
        );
        M<T> test_mat_2_4(
            TestBase::bundle,
            {{static_cast<T>(0.), static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-2.)},
             {static_cast<T>(10.), static_cast<T>(0.), static_cast<T>(12.), static_cast<T>(12.)}}
        );
        M<T> test_mat_4_3(
            TestBase::bundle,
            {{static_cast<T>(1.), static_cast<T>(2.), static_cast<T>(3.)},
             {static_cast<T>(4.), static_cast<T>(5.), static_cast<T>(6.)},
             {static_cast<T>(7.), static_cast<T>(8.), static_cast<T>(9.)},
             {static_cast<T>(10.), static_cast<T>(11.), static_cast<T>(12.)}}
        );

        // Copy to empty
        test_mat_empty = test_mat_4_3;
        ASSERT_EQ(test_mat_empty.rows(), test_mat_4_3.rows());
        ASSERT_EQ(test_mat_empty.cols(), test_mat_4_3.cols());
        ASSERT_MATRIX_EQ(test_mat_empty, test_mat_4_3);

        // Copy to populated
        test_mat_2_2 = test_mat_4_3;
        ASSERT_EQ(test_mat_2_2.rows(), test_mat_4_3.rows());
        ASSERT_EQ(test_mat_2_2.cols(), test_mat_4_3.cols());
        ASSERT_MATRIX_EQ(test_mat_2_2, test_mat_4_3);

        // Reassignment
        test_mat_2_2 = test_mat_2_4;
        ASSERT_EQ(test_mat_2_2.rows(), test_mat_2_4.rows());
        ASSERT_EQ(test_mat_2_2.cols(), test_mat_2_4.cols());
        ASSERT_MATRIX_EQ(test_mat_2_2, test_mat_2_4);

        // Transitive assignment
        test_mat_empty = test_mat_2_2;
        ASSERT_EQ(test_mat_empty.rows(), test_mat_2_4.rows());
        ASSERT_EQ(test_mat_empty.cols(), test_mat_2_4.cols());
        ASSERT_MATRIX_EQ(test_mat_empty, test_mat_2_4);

        // Self-assignment
        test_mat_4_3 = test_mat_4_3;
        ASSERT_EQ(test_mat_4_3.rows(), 4);
        ASSERT_EQ(test_mat_4_3.cols(), 3);
        ASSERT_EQ(test_mat_4_3.get_elem(0, 0).get_scalar(), static_cast<T>(1.));
        ASSERT_EQ(test_mat_4_3.get_elem(0, 1).get_scalar(), static_cast<T>(2.));
        ASSERT_EQ(test_mat_4_3.get_elem(0, 2).get_scalar(), static_cast<T>(3.));
        ASSERT_EQ(test_mat_4_3.get_elem(1, 0).get_scalar(), static_cast<T>(4.));
        ASSERT_EQ(test_mat_4_3.get_elem(1, 1).get_scalar(), static_cast<T>(5.));
        ASSERT_EQ(test_mat_4_3.get_elem(1, 2).get_scalar(), static_cast<T>(6.));
        ASSERT_EQ(test_mat_4_3.get_elem(2, 0).get_scalar(), static_cast<T>(7.));
        ASSERT_EQ(test_mat_4_3.get_elem(2, 1).get_scalar(), static_cast<T>(8.));
        ASSERT_EQ(test_mat_4_3.get_elem(2, 2).get_scalar(), static_cast<T>(9.));
        ASSERT_EQ(test_mat_4_3.get_elem(3, 0).get_scalar(), static_cast<T>(10.));
        ASSERT_EQ(test_mat_4_3.get_elem(3, 1).get_scalar(), static_cast<T>(11.));
        ASSERT_EQ(test_mat_4_3.get_elem(3, 2).get_scalar(), static_cast<T>(12.));

    }

    template <typename T>
    void TestCopyConstructor() {

        M<T> test_mat_2_4(
            TestBase::bundle,
            {{static_cast<T>(0.), static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-2.)},
             {static_cast<T>(10.), static_cast<T>(0.), static_cast<T>(12.), static_cast<T>(12.)}}
        );

        M<T> test_mat_copied(test_mat_2_4);
        ASSERT_EQ(test_mat_copied.rows(), 2);
        ASSERT_EQ(test_mat_copied.cols(), 4);
        ASSERT_EQ(test_mat_copied.get_elem(0, 0).get_scalar(), static_cast<T>(0.));
        ASSERT_EQ(test_mat_copied.get_elem(0, 1).get_scalar(), static_cast<T>(-1.));
        ASSERT_EQ(test_mat_copied.get_elem(0, 2).get_scalar(), static_cast<T>(-2.));
        ASSERT_EQ(test_mat_copied.get_elem(0, 3).get_scalar(), static_cast<T>(-2.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 0).get_scalar(), static_cast<T>(10.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 1).get_scalar(), static_cast<T>(0.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 2).get_scalar(), static_cast<T>(12.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 3).get_scalar(), static_cast<T>(12.));

    }

    template <typename T>
    void TestStaticCreation() {

        constexpr int m_zero(15);
        constexpr int n_zero(17);
        M<T> test_zero(M<T>::Zero(TestBase::bundle, m_zero, n_zero));
        ASSERT_EQ(test_zero.rows(), m_zero);
        ASSERT_EQ(test_zero.cols(), n_zero);
        for (int i=0; i<m_zero; ++i) {
            for (int j=0; j<n_zero; ++j) {
                ASSERT_EQ(test_zero.get_elem(i, j).get_scalar(), static_cast<T>(0.));
            }
        }

        constexpr int m_one(32);
        constexpr int n_one(13);
        M<T> test_ones(M<T>::Ones(TestBase::bundle, m_one, n_one));
        ASSERT_EQ(test_ones.rows(), m_one);
        ASSERT_EQ(test_ones.cols(), n_one);
        for (int i=0; i<m_one; ++i) {
            for (int j=0; j<n_one; ++j) {
                ASSERT_EQ(test_ones.get_elem(i, j).get_scalar(), static_cast<T>(1.));
            }
        }

        constexpr int m_identity(40);
        constexpr int n_identity(20);
        M<T> test_identity(M<T>::Identity(TestBase::bundle, m_identity, n_identity));
        ASSERT_EQ(test_identity.rows(), m_identity);
        ASSERT_EQ(test_identity.cols(), n_identity);
        for (int i=0; i<m_identity; ++i) {
            for (int j=0; j<n_identity; ++j) {
                if (i == j) {
                    ASSERT_EQ(test_identity.get_elem(i, j).get_scalar(), static_cast<T>(1.));
                } else {
                    ASSERT_EQ(test_identity.get_elem(i, j).get_scalar(), static_cast<T>(0.));
                }
            }
        }

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from 5 adjacent above and below)
        constexpr int m_rand(40);
        constexpr int n_rand(40);
        M<T> test_rand(M<T>::Random(TestBase::bundle, m_rand, n_rand));
        ASSERT_EQ(test_rand.rows(), m_rand);
        ASSERT_EQ(test_rand.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    ((test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i-1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i+1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i, j-1).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i, j+1).get_scalar()))
                );
            }
        }

    }

    template <typename T>
    void TestCol() {

        const M<T> const_mat(
            TestBase::bundle, 
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );
        M<T> mat(const_mat);

        // Test Col access
        for (int j=0; j<3; ++j) {
            typename M<T>::Col col(mat.get_col(j));
            for (int i=0; i<4; ++i) {
                ASSERT_EQ(col.get_elem(i), const_mat.get_elem(i, j));
            }
        }

    }

    void TestBadCol() {

        const int m(4);
        const int n(3);
        M<double> mat(
            TestBase::bundle, 
            {{1, 2, 3},
             {4, 5, 6},
             {7, 8, 9},
             {10, 11, 12}}
        );

        // Test bad col
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_col(-1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_col(n); });

        // Test bad access in valid col
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_col(0).get_elem(-1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_col(0).get_elem(m); });

    }

    template <typename T>
    void TestScale() {

        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(-8), static_cast<T>(0.8), static_cast<T>(-0.6)},
             {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(-3), static_cast<T>(-2), static_cast<T>(-1)},
             {static_cast<T>(10), static_cast<T>(100), static_cast<T>(1000)}}
        );
        M<T> mat_scaled_mult(mat*Scalar<T>(static_cast<T>(4)));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    mat_scaled_mult.get_elem(i, j).get_scalar(),
                    static_cast<T>(4)*mat.get_elem(i, j).get_scalar()
                );
            }
        }
        M<T> mat_scaled_div(mat/Scalar<T>(static_cast<T>(10)));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    mat_scaled_div.get_elem(i, j).get_scalar(),
                    (static_cast<T>(1)/static_cast<T>(10))*mat.get_elem(i, j).get_scalar()
                );
            }
        }

    }

    template <typename T>
    void TestScaleAssignment() {

        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(-8), static_cast<T>(0.8), static_cast<T>(-0.6)},
             {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(-3), static_cast<T>(-2), static_cast<T>(-1)},
             {static_cast<T>(10), static_cast<T>(100), static_cast<T>(1000)}}
        );
        M<T> temp_mat_1(mat);
        M<T> temp_mat_2(mat);
        temp_mat_1 *= Scalar<T>(static_cast<T>(4));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    temp_mat_1.get_elem(i, j).get_scalar(),
                    static_cast<T>(4)*mat.get_elem(i, j).get_scalar()
                );
            }
        }
        temp_mat_2 /= Scalar<T>(static_cast<T>(10));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    temp_mat_2.get_elem(i, j).get_scalar(),
                    (static_cast<T>(1)/static_cast<T>(10))*mat.get_elem(i, j).get_scalar()
                );
            }
        }

    }

    template <typename T>
    void TestMaxMagElem() {

        M<T> mat_all_diff(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(3), static_cast<T>(2), static_cast<T>(1)}}
        );
        ASSERT_EQ(mat_all_diff.get_max_mag_elem().get_scalar(), static_cast<T>(9));

        M<T> mat_all_same(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        ASSERT_EQ(mat_all_same.get_max_mag_elem().get_scalar(), static_cast<T>(1));

        M<T> mat_pos_neg(
            TestBase::bundle,
            {{static_cast<T>(-1), static_cast<T>(-9), static_cast<T>(8)},
             {static_cast<T>(1), static_cast<T>(-14), static_cast<T>(10)}}
        );
        ASSERT_EQ(mat_pos_neg.get_max_mag_elem().get_scalar(), static_cast<T>(14));

    }

    template <typename T>
    void TestNormalizeMagnitude() {

        M<T> mat_has_zeros(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(4), static_cast<T>(0), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(3), static_cast<T>(0), static_cast<T>(1)}}
        );
        M<T> temp_mat_has_zeros(mat_has_zeros);
        temp_mat_has_zeros.normalize_magnitude();
        ASSERT_MATRIX_NEAR(
            temp_mat_has_zeros,
            mat_has_zeros/Scalar<T>(static_cast<T>(9)),
            Tol<T>::roundoff_T()
        );

        M<T> mat_all_same(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        M<T> temp_mat_all_same(mat_all_same);
        temp_mat_all_same.normalize_magnitude();
        ASSERT_MATRIX_NEAR(
            temp_mat_all_same,
            mat_all_same,
            Tol<T>::roundoff_T()
        );

        M<T> mat_pos_neg(
            TestBase::bundle,
            {{static_cast<T>(-1), static_cast<T>(-9), static_cast<T>(8)},
             {static_cast<T>(1), static_cast<T>(-14), static_cast<T>(10)}}
        );
        M<T> temp_mat_pos_neg(mat_pos_neg);
        temp_mat_pos_neg.normalize_magnitude();
        ASSERT_MATRIX_NEAR(
            temp_mat_pos_neg,
            mat_pos_neg/Scalar<T>(static_cast<T>(14)),
            Tol<T>::roundoff_T()
        );

    }

    template <typename T>
    void TestMatVec() {

        // Test manually
        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(3), static_cast<T>(2), static_cast<T>(1)}}
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1), static_cast<T>(4), static_cast<T>(7), static_cast<T>(3)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(2), static_cast<T>(5), static_cast<T>(8), static_cast<T>(2)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(3), static_cast<T>(6), static_cast<T>(9), static_cast<T>(1)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01)}
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1.23), static_cast<T>(4.56), static_cast<T>(7.89), static_cast<T>(3.21)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

        // Test random
        const int m_rand(3);
        const int n_rand(4);
        M<T> rand_mat(M<T>::Random(TestBase::bundle, m_rand, n_rand));
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)}
            ),
            Vector<T>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 0).get_scalar(),
                 rand_mat.get_elem(1, 0).get_scalar(),
                 rand_mat.get_elem(2, 0).get_scalar()}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
            ),
            Vector<T>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 1).get_scalar(),
                 rand_mat.get_elem(1, 1).get_scalar(),
                 rand_mat.get_elem(2, 1).get_scalar()}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
            ),
            Vector<T>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 2).get_scalar(),
                 rand_mat.get_elem(1, 2).get_scalar(),
                 rand_mat.get_elem(2, 2).get_scalar()}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
            ),
            Vector<T>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 3).get_scalar(),
                 rand_mat.get_elem(1, 3).get_scalar(),
                 rand_mat.get_elem(2, 3).get_scalar()}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01), static_cast<T>(0.001)}
            ),
            Vector<T>(
                TestBase::bundle,
                {(static_cast<T>(1)*rand_mat.get_elem(0, 0).get_scalar() +
                  static_cast<T>(0.1)*rand_mat.get_elem(0, 1).get_scalar() +
                  static_cast<T>(0.01)*rand_mat.get_elem(0, 2).get_scalar() +
                  static_cast<T>(0.001)*rand_mat.get_elem(0, 3).get_scalar()),
                 (static_cast<T>(1)*rand_mat.get_elem(1, 0).get_scalar() +
                  static_cast<T>(0.1)*rand_mat.get_elem(1, 1).get_scalar() +
                  static_cast<T>(0.01)*rand_mat.get_elem(1, 2).get_scalar()+
                  static_cast<T>(0.001)*rand_mat.get_elem(1, 3).get_scalar()),
                 (static_cast<T>(1)*rand_mat.get_elem(2, 0).get_scalar() +
                  static_cast<T>(0.1)*rand_mat.get_elem(2, 1).get_scalar() +
                  static_cast<T>(0.01)*rand_mat.get_elem(2, 2).get_scalar()+
                  static_cast<T>(0.001)*rand_mat.get_elem(2, 3).get_scalar())}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

    }

    template <typename T>
    void TestBadMatVec() {

        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );

        Vector<T> vec_too_small(
            TestBase::bundle,
            {static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_too_small; }
        );

        Vector<T> vec_too_large(
            TestBase::bundle, 
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_too_large; }
        );

        Vector<T> vec_matches_wrong_dimension(
            TestBase::bundle,
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_matches_wrong_dimension; }
        );

    }

    template <typename T>
    void TestTransposeMatVec() {

        // Test manually
        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)}}
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(9), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {static_cast<T>(1.59), static_cast<T>(2.61), static_cast<T>(3.72), static_cast<T>(4.83)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

        // Test random
        const int m_rand(3);
        const int n_rand(2);
        M<T> rand_mat(M<T>::Random(TestBase::bundle, m_rand, n_rand));
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 0).get_scalar(),
                 rand_mat.get_elem(0, 1).get_scalar()}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {rand_mat.get_elem(1, 0).get_scalar(),
                 rand_mat.get_elem(1, 1).get_scalar()}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {rand_mat.get_elem(2, 0).get_scalar(),
                 rand_mat.get_elem(2, 1).get_scalar()}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<T>(
                    TestBase::bundle,
                    {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01)}
                )
            ),
            Vector<T>(
                TestBase::bundle,
                {(static_cast<T>(1)*rand_mat.get_elem(0, 0).get_scalar() +
                  static_cast<T>(0.1)*rand_mat.get_elem(1, 0).get_scalar() +
                  static_cast<T>(0.01)*rand_mat.get_elem(2, 0).get_scalar()),
                 (static_cast<T>(1)*rand_mat.get_elem(0, 1).get_scalar() +
                  static_cast<T>(0.1)*rand_mat.get_elem(1, 1).get_scalar() +
                  static_cast<T>(0.01)*rand_mat.get_elem(2, 1).get_scalar())}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

    }

    template <typename T>
    void TestBadTransposeMatVec() {

        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );

        Vector<T> vec_too_small(
            TestBase::bundle,
            {static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_too_small); }
        );

        Vector<T> vec_too_large(
            TestBase::bundle,
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_too_large); }
        );

        Vector<T> vec_matches_wrong_dimension(
            TestBase::bundle,
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_matches_wrong_dimension); }
        );

    }

    template <typename T>
    void TestTranspose() {

        // Test manually
        constexpr int m_manual(4);
        constexpr int n_manual(3);
        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );
        M<T> mat_transposed(mat.transpose());
        ASSERT_EQ(mat_transposed.rows(), m_manual);
        ASSERT_EQ(mat_transposed.cols(), n_manual);
        for (int i=0; i<m_manual; ++i) {
            for (int j=0; j<n_manual; ++j) {
                ASSERT_EQ(
                    mat_transposed.get_elem(i, j).get_scalar(),
                    mat.get_elem(j, i).get_scalar()
                );
            }
        }
        M<T> test(
            TestBase::bundle, 
            {{static_cast<T>(1), static_cast<T>(5), static_cast<T>(9)},
             {static_cast<T>(2), static_cast<T>(6), static_cast<T>(10)},
             {static_cast<T>(3), static_cast<T>(7), static_cast<T>(11)},
             {static_cast<T>(4), static_cast<T>(8), static_cast<T>(12)}}
        );
        ASSERT_MATRIX_EQ(mat_transposed, test);

        // Test random
        constexpr int m_rand(12);
        constexpr int n_rand(17);
        M<T> mat_rand(M<T>::Random(TestBase::bundle, m_rand, n_rand));
        M<T> mat_rand_transposed(mat_rand.transpose());
        ASSERT_EQ(mat_rand_transposed.rows(), n_rand);
        ASSERT_EQ(mat_rand_transposed.cols(), m_rand);
        for (int i=0; i<n_rand; ++i) {
            for (int j=0; j<m_rand; ++j) {
                ASSERT_EQ(
                    mat_rand_transposed.get_elem(i, j).get_scalar(),
                    mat_rand.get_elem(j, i).get_scalar()
                );
            }
        }

    }

    template <typename T>
    void TestMatMat() {

        M<T> mat1(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(4)},
             {static_cast<T>(-1), static_cast<T>(-4)},
             {static_cast<T>(-3), static_cast<T>(-2)},
             {static_cast<T>(10), static_cast<T>(100)}}
        );
        M<T> mat2(
            TestBase::bundle,
            {{static_cast<T>(7), static_cast<T>(2), static_cast<T>(-3)},
             {static_cast<T>(10), static_cast<T>(-4), static_cast<T>(-3)}}
        );
        M<T> test_mat(
            TestBase::bundle,
            {{static_cast<T>(47), static_cast<T>(-14), static_cast<T>(-15)},
             {static_cast<T>(-47), static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(-41), static_cast<T>(2), static_cast<T>(15)},
             {static_cast<T>(1070), static_cast<T>(-380), static_cast<T>(-330)}}
        );
        ASSERT_MATRIX_EQ(mat1*mat2, test_mat);
        ASSERT_MATRIX_EQ(mat1*M<T>::Identity(TestBase::bundle, 2, 2), mat1);
        ASSERT_MATRIX_EQ(mat2*M<T>::Identity(TestBase::bundle, 3, 3), mat2);

    }

    template <typename T>
    void TestBadMatMat() {

        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(4)},
             {static_cast<T>(-1), static_cast<T>(-4)},
             {static_cast<T>(-3), static_cast<T>(-2)},
             {static_cast<T>(10), static_cast<T>(100)}}
        );

        M<T> mat_too_small(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat*mat_too_small; });

        M<T> mat_too_big(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat*mat_too_big; });

        M<T> mat_matches_other_dim(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat*mat_matches_other_dim; });

    }

    template <typename T>
    void TestAddSub() {

        M<T> mat1(
            TestBase::bundle,
            {{static_cast<T>(47), static_cast<T>(-14), static_cast<T>(-15)},
             {static_cast<T>(-47), static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(-41), static_cast<T>(2), static_cast<T>(15)},
             {static_cast<T>(10), static_cast<T>(-38), static_cast<T>(-33)}}
        );

        M<T> mat2(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(-4), static_cast<T>(-5), static_cast<T>(-6)},
             {static_cast<T>(-7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(10), static_cast<T>(-11), static_cast<T>(-12)}}
        );

        M<T> test_mat_add(
            TestBase::bundle,
            {{static_cast<T>(48), static_cast<T>(-12), static_cast<T>(-12)},
             {static_cast<T>(-51), static_cast<T>(9), static_cast<T>(9)},
             {static_cast<T>(-48), static_cast<T>(10), static_cast<T>(24)},
             {static_cast<T>(20), static_cast<T>(-49), static_cast<T>(-45)}}
        );
        ASSERT_MATRIX_EQ(mat1+mat2, test_mat_add);

        M<T> test_mat_sub_1(
            TestBase::bundle,
            {{static_cast<T>(46), static_cast<T>(-16), static_cast<T>(-18)},
             {static_cast<T>(-43), static_cast<T>(19), static_cast<T>(21)},
             {static_cast<T>(-34), static_cast<T>(-6), static_cast<T>(6)},
             {static_cast<T>(0), static_cast<T>(-27), static_cast<T>(-21)}}
        );
        ASSERT_MATRIX_EQ(mat1-mat2, test_mat_sub_1);

        M<T> test_mat_sub_2(
            TestBase::bundle,
            {{static_cast<T>(-46), static_cast<T>(16), static_cast<T>(18)},
             {static_cast<T>(43), static_cast<T>(-19), static_cast<T>(-21)},
             {static_cast<T>(34), static_cast<T>(6), static_cast<T>(-6)},
             {static_cast<T>(0), static_cast<T>(27), static_cast<T>(21)}}
        );
        ASSERT_MATRIX_EQ(mat2-mat1, test_mat_sub_2);

        ASSERT_MATRIX_EQ(mat1-mat1, M<T>::Zero(TestBase::bundle, 4, 3));
        ASSERT_MATRIX_EQ(mat2-mat2, M<T>::Zero(TestBase::bundle, 4, 3));

    }

    template <typename T>
    void TestBadAddSub() {

        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(47), static_cast<T>(-14), static_cast<T>(-15)},
             {static_cast<T>(-47), static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(-41), static_cast<T>(2), static_cast<T>(15)},
             {static_cast<T>(10), static_cast<T>(-38), static_cast<T>(-33)}}
        );

        M<T> mat_too_few_cols(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_few_cols; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_few_cols; });

        M<T> mat_too_many_cols(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_many_cols; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_many_cols; });

        M<T> mat_too_few_rows(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_few_rows; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_few_rows; });

        M<T> mat_too_many_rows(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_many_rows; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_many_rows; });

    }

    template <typename T>
    void TestNorm() {

        M<T> mat1(
            TestBase::bundle,
            {{static_cast<T>(18), static_cast<T>(5)},
             {static_cast<T>(-3), static_cast<T>(6)},
             {static_cast<T>(9), static_cast<T>(-9)},
             {static_cast<T>(4), static_cast<T>(-2)}}
        );
        ASSERT_EQ(mat1.norm().get_scalar(), static_cast<T>(24));

        M<T> mat2(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1)},
             {static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(-1)}}
        );
        ASSERT_EQ(mat2.norm().get_scalar(), static_cast<T>(3));

    }

    void TestCast() {
        
        constexpr int m(10);
        constexpr int n(15);

        M<double> mat_dbl(M<double>::Random(TestBase::bundle, m, n));

        M<double> dbl_to_dbl(mat_dbl.template cast<double>());
        ASSERT_MATRIX_EQ(dbl_to_dbl, mat_dbl);

        M<float> dbl_to_sgl(mat_dbl.template cast<float>());
        ASSERT_EQ(dbl_to_sgl.rows(), m);
        ASSERT_EQ(dbl_to_sgl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    dbl_to_sgl.get_elem(i, j).get_scalar(),
                    static_cast<float>(mat_dbl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<float>(mat_dbl.get_elem(i, j).get_scalar()))*
                        Tol<float>::roundoff_T()
                );
            }
        }

        M<__half> dbl_to_hlf(mat_dbl.template cast<__half>());
        ASSERT_EQ(dbl_to_hlf.rows(), m);
        ASSERT_EQ(dbl_to_hlf.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    dbl_to_hlf.get_elem(i, j).get_scalar(),
                    static_cast<__half>(mat_dbl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<__half>(mat_dbl.get_elem(i, j).get_scalar()))*
                        Tol<__half>::roundoff_T()
                );
            }
        }

        M<float> mat_sgl(M<float>::Random(TestBase::bundle, m, n));

        M<float> sgl_to_sgl(mat_sgl.template cast<float>());
        ASSERT_MATRIX_EQ(sgl_to_sgl, mat_sgl);
    
        M<double> sgl_to_dbl(mat_sgl.template cast<double>());
        ASSERT_EQ(sgl_to_dbl.rows(), m);
        ASSERT_EQ(sgl_to_dbl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    sgl_to_dbl.get_elem(i, j).get_scalar(),
                    static_cast<double>(mat_sgl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<double>(mat_sgl.get_elem(i, j).get_scalar()))*
                        static_cast<double>(Tol<float>::roundoff_T())
                );
            }
        }

        M<__half> sgl_to_hlf(mat_sgl.template cast<__half>());
        ASSERT_EQ(sgl_to_hlf.rows(), m);
        ASSERT_EQ(sgl_to_hlf.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    sgl_to_hlf.get_elem(i, j).get_scalar(),
                    static_cast<__half>(mat_sgl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<__half>(mat_sgl.get_elem(i, j).get_scalar()))*
                        Tol<__half>::roundoff_T()
                );
            }
        }

        M<__half> mat_hlf(M<__half>::Random(TestBase::bundle, m, n));

        M<__half> hlf_to_hlf(mat_hlf.template cast<__half>());
        ASSERT_MATRIX_EQ(hlf_to_hlf, mat_hlf);

        M<float> hlf_to_sgl(mat_hlf.template cast<float>());
        ASSERT_EQ(hlf_to_sgl.rows(), m);
        ASSERT_EQ(hlf_to_sgl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    hlf_to_sgl.get_elem(i, j).get_scalar(),
                    static_cast<float>(mat_hlf.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<double>(mat_hlf.get_elem(i, j).get_scalar()))*
                        static_cast<float>(Tol<__half>::roundoff_T())
                );
            }
        }

        M<double> hlf_to_dbl(mat_hlf.template cast<double>());
        ASSERT_EQ(hlf_to_dbl.rows(), m);
        ASSERT_EQ(hlf_to_dbl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    hlf_to_dbl.get_elem(i, j).get_scalar(),
                    static_cast<double>(mat_hlf.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<double>(mat_hlf.get_elem(i, j).get_scalar()))*
                        static_cast<double>(Tol<__half>::roundoff_T())
                );
            }
        }

    }

    void TestBadCast() {

        auto try_bad_cast = []() {
            const int m(20);
            const int n(30);
            M<double> mat(TestBase::bundle, m, n);
            mat.template cast<int>();
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_bad_cast);

    }

};

template <template <typename> typename M>
class Matrix_Substitution_Test: public TestBase
{
public:

    template <typename T>
    void TestBackwardSubstitution() {

        const double approx_U_tri_cond_number_upbound(2.5);
        constexpr int n(90);

        M<T> U_tri(
            read_matrixCSV<M, T>(TestBase::bundle, solve_matrix_dir / fs::path("U_tri_90.csv"))
        );
        Vector<T> x_tri(
            read_matrixCSV<Vector, T>(TestBase::bundle, solve_matrix_dir / fs::path("x_tri_90.csv"))
        );
        Vector<T> Ub_tri(
            read_matrixCSV<Vector, T>(TestBase::bundle, solve_matrix_dir / fs::path("Ub_tri_90.csv"))
        );
    
        Vector<T> test_soln(U_tri.back_sub(Ub_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            Tol<T>::substitution_tol_T(approx_U_tri_cond_number_upbound, 90)
        );

    }

    template <typename T>
    void TestForwardSubstitution() {

        const double approx_L_tri_cond_number_upbound(2.5);
        const int n(90);

        M<T> L_tri(
            read_matrixCSV<M, T>(TestBase::bundle, solve_matrix_dir / fs::path("L_tri_90.csv"))
        );
        Vector<T> x_tri(
            read_matrixCSV<Vector, T>(TestBase::bundle, solve_matrix_dir / fs::path("x_tri_90.csv"))
        );
        Vector<T> Lb_tri(
            read_matrixCSV<Vector, T>(TestBase::bundle, solve_matrix_dir / fs::path("Lb_tri_90.csv"))
        );
    
        Vector<T> test_soln(L_tri.frwd_sub(Lb_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            Tol<T>::substitution_tol_T(approx_L_tri_cond_number_upbound, 90)
        );

    }

};