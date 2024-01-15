#include "../test.h"

class Matrix_Test: public TestBase
{
protected:

    template <template <typename> typename M, typename T>
    void TestCoeffAccess_Base() {
        
        constexpr int m(24);
        constexpr int n(12);
        M<T> test_mat(*handle_ptr, m, n);

        T elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                test_mat.set_elem(i, j, elem);
                elem += static_cast<T>(1);
            }
        }

        T test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat.get_elem(i, j), test_elem);
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 0 row as all -1
        for (int j=0; j<n; ++j) { test_mat.set_elem(0, j, static_cast<T>(-1.)); }

        // Test matches modified matrix
        test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) { ASSERT_EQ(test_mat.get_elem(i, j), static_cast<T>(-1.)); }
                else { ASSERT_EQ(test_mat.get_elem(i, j), test_elem); }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 4 row as decreasing by -1 from -1
        T row_5_elem = static_cast<T>(-1.);
        for (int j=0; j<n; ++j) {
            test_mat.set_elem(4, j, row_5_elem);
            row_5_elem += static_cast<T>(-1.);
        }

        // Test matches modified matrix
        test_elem = static_cast<T>(1);
        T row_5_test_elem = static_cast<T>(-1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) { ASSERT_EQ(test_mat.get_elem(i, j), static_cast<T>(-1.)); }
                else if (i == 4) { ASSERT_EQ(test_mat.get_elem(i, j), row_5_test_elem);
                                   row_5_test_elem += static_cast<T>(-1.);}
                else { ASSERT_EQ(test_mat.get_elem(i, j), test_elem); }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 2 col as incresing by 1 from -5
        T coL_3_elem = static_cast<T>(-5.);
        for (int i=0; i<m; ++i) {
            test_mat.set_elem(i, 2, coL_3_elem);
            coL_3_elem += static_cast<T>(1.);
        }

        // Test matches modified matrix
        test_elem = static_cast<T>(1);
        row_5_test_elem = static_cast<T>(-1.);
        T coL_3_test_elem = static_cast<T>(-5.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (j == 2) {
                    ASSERT_EQ(test_mat.get_elem(i, j), coL_3_test_elem);
                    coL_3_test_elem += static_cast<T>(1.);
                } else if (i == 0) {
                    ASSERT_EQ(test_mat.get_elem(i, j), static_cast<T>(-1.));
                } else if (i == 4) {
                    ASSERT_EQ(test_mat.get_elem(i, j), row_5_test_elem);
                    row_5_test_elem += static_cast<T>(-1.);
                    if (j == 1) { row_5_test_elem += static_cast<T>(-1.); }
                } else {
                    ASSERT_EQ(test_mat.get_elem(i, j), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

    }

    template <template <typename> typename M>
    void TestBadCoeffAccess_Base() {
        
        constexpr int m(24);
        constexpr int n(12);
        M<double> test_mat(M<double>::Random(*handle_ptr, m, n));

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(0, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(0, n); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(-1, 0); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_mat.get_elem(m, 0); });

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { test_mat.set_elem(0, -1, 0.); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { test_mat.set_elem(0, n, 0.); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { test_mat.set_elem(-1, 0, 0.); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { test_mat.set_elem(m, 0, 0.); });

    }

    template <template <typename> typename M, typename T>
    void TestPropertyAccess_Base() {

        constexpr int m_1(62);
        constexpr int n_1(3);
        M<T> mat_1(*handle_ptr, m_1, n_1);
        ASSERT_EQ(mat_1.rows(), m_1);
        ASSERT_EQ(mat_1.cols(), n_1);

        constexpr int m_2(15);
        constexpr int n_2(90);
        M<T> mat_2(*handle_ptr, m_2, n_2);
        ASSERT_EQ(mat_2.rows(), m_2);
        ASSERT_EQ(mat_2.cols(), n_2);

        constexpr int m_3(20);
        constexpr int n_3(20);
        M<T> mat_3(*handle_ptr, m_3, n_3);
        ASSERT_EQ(mat_3.rows(), m_3);
        ASSERT_EQ(mat_3.cols(), n_3);

    }

    template <template <typename> typename M, typename T>
    void TestConstruction_Base() {

        M<T> test_mat_empty(*handle_ptr);
        ASSERT_EQ(test_mat_empty.rows(), 0);
        ASSERT_EQ(test_mat_empty.cols(), 0);

        constexpr int m(12);
        M<T> test_mat_square(*handle_ptr, m, m);
        ASSERT_EQ(test_mat_square.rows(), m);
        ASSERT_EQ(test_mat_square.cols(), m);

        constexpr int n(33);
        M<T> test_mat_wide(*handle_ptr, m, n);
        ASSERT_EQ(test_mat_wide.rows(), m);
        ASSERT_EQ(test_mat_wide.cols(), n);

        M<T> test_mat_tall(*handle_ptr, n, m);
        ASSERT_EQ(test_mat_tall.rows(), n);
        ASSERT_EQ(test_mat_tall.cols(), m);

    }

    template <template <typename> typename M, typename T>
    void TestListInitialization_Base() {

        M<T> test_mat_0_0 (*handle_ptr, {});
        ASSERT_EQ(test_mat_0_0.rows(), 0);
        ASSERT_EQ(test_mat_0_0.cols(), 0);

        M<T> test_mat_0_1 (*handle_ptr, {{}});
        ASSERT_EQ(test_mat_0_1.rows(), 1);
        ASSERT_EQ(test_mat_0_1.cols(), 0);

        M<T> test_mat_1(
            *handle_ptr,
            {{static_cast<T>(5.), static_cast<T>(3.), static_cast<T>(27.)},
             {static_cast<T>(88.), static_cast<T>(-4.), static_cast<T>(-6.)},
             {static_cast<T>(100.), static_cast<T>(12.), static_cast<T>(2.)}}
        );
        ASSERT_EQ(test_mat_1.rows(), 3);
        ASSERT_EQ(test_mat_1.cols(), 3);
        ASSERT_EQ(test_mat_1.get_elem(0, 0), static_cast<T>(5.));
        ASSERT_EQ(test_mat_1.get_elem(0, 1), static_cast<T>(3.));
        ASSERT_EQ(test_mat_1.get_elem(0, 2), static_cast<T>(27.));
        ASSERT_EQ(test_mat_1.get_elem(1, 0), static_cast<T>(88.));
        ASSERT_EQ(test_mat_1.get_elem(1, 1), static_cast<T>(-4.));
        ASSERT_EQ(test_mat_1.get_elem(1, 2), static_cast<T>(-6.));
        ASSERT_EQ(test_mat_1.get_elem(2, 0), static_cast<T>(100.));
        ASSERT_EQ(test_mat_1.get_elem(2, 1), static_cast<T>(12.));
        ASSERT_EQ(test_mat_1.get_elem(2, 2), static_cast<T>(2.));

        M<T> test_mat_wide(
            *handle_ptr,
            {{static_cast<T>(7.), static_cast<T>(5.), static_cast<T>(3.)},
             {static_cast<T>(1.), static_cast<T>(6.), static_cast<T>(2.)}}
        );
        ASSERT_EQ(test_mat_wide.rows(), 2);
        ASSERT_EQ(test_mat_wide.cols(), 3);
        ASSERT_EQ(test_mat_wide.get_elem(0, 0), static_cast<T>(7.));
        ASSERT_EQ(test_mat_wide.get_elem(0, 1), static_cast<T>(5.));
        ASSERT_EQ(test_mat_wide.get_elem(0, 2), static_cast<T>(3.));
        ASSERT_EQ(test_mat_wide.get_elem(1, 0), static_cast<T>(1.));
        ASSERT_EQ(test_mat_wide.get_elem(1, 1), static_cast<T>(6.));
        ASSERT_EQ(test_mat_wide.get_elem(1, 2), static_cast<T>(2.));
        
        M<T> test_mat_tall(
            *handle_ptr,
            {{static_cast<T>(7.), static_cast<T>(5.)},
             {static_cast<T>(1.), static_cast<T>(6.)},
             {static_cast<T>(3.), static_cast<T>(2.)},
             {static_cast<T>(43.), static_cast<T>(9.)}}
        );
        ASSERT_EQ(test_mat_tall.rows(), 4);
        ASSERT_EQ(test_mat_tall.cols(), 2);
        ASSERT_EQ(test_mat_tall.get_elem(0, 0), static_cast<T>(7.));
        ASSERT_EQ(test_mat_tall.get_elem(0, 1), static_cast<T>(5.));
        ASSERT_EQ(test_mat_tall.get_elem(1, 0), static_cast<T>(1.));
        ASSERT_EQ(test_mat_tall.get_elem(1, 1), static_cast<T>(6.));
        ASSERT_EQ(test_mat_tall.get_elem(2, 0), static_cast<T>(3.));
        ASSERT_EQ(test_mat_tall.get_elem(2, 1), static_cast<T>(2.));
        ASSERT_EQ(test_mat_tall.get_elem(3, 0), static_cast<T>(43.));
        ASSERT_EQ(test_mat_tall.get_elem(3, 1), static_cast<T>(9.));

    }

    template <template <typename> typename M>
    void TestBadListInitialization_Base() {

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(*handle_ptr, {{1, 2, 3, 4}, {1, 2, 3}}); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(*handle_ptr, {{1, 2}, {1, 2, 3}}); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(*handle_ptr, {{1, 2}, {}}); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { M<double> mat(*handle_ptr, {{}, {1, 2, 3}}); }
        );

    }

    template <template <typename> typename M, typename T>
    void TestCopyAssignment_Base() {

        M<T> test_mat_empty(*handle_ptr, {{}});
        M<T> test_mat_2_2(
            *handle_ptr,
            {{static_cast<T>(-3.), static_cast<T>(0.)},
             {static_cast<T>(1.), static_cast<T>(10.)}}
        );
        M<T> test_mat_2_4(
            *handle_ptr,
            {{static_cast<T>(0.), static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-2.)},
             {static_cast<T>(10.), static_cast<T>(0.), static_cast<T>(12.), static_cast<T>(12.)}}
        );
        M<T> test_mat_4_3(
            *handle_ptr,
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
        ASSERT_EQ(test_mat_4_3.get_elem(0, 0), static_cast<T>(1.));
        ASSERT_EQ(test_mat_4_3.get_elem(0, 1), static_cast<T>(2.));
        ASSERT_EQ(test_mat_4_3.get_elem(0, 2), static_cast<T>(3.));
        ASSERT_EQ(test_mat_4_3.get_elem(1, 0), static_cast<T>(4.));
        ASSERT_EQ(test_mat_4_3.get_elem(1, 1), static_cast<T>(5.));
        ASSERT_EQ(test_mat_4_3.get_elem(1, 2), static_cast<T>(6.));
        ASSERT_EQ(test_mat_4_3.get_elem(2, 0), static_cast<T>(7.));
        ASSERT_EQ(test_mat_4_3.get_elem(2, 1), static_cast<T>(8.));
        ASSERT_EQ(test_mat_4_3.get_elem(2, 2), static_cast<T>(9.));
        ASSERT_EQ(test_mat_4_3.get_elem(3, 0), static_cast<T>(10.));
        ASSERT_EQ(test_mat_4_3.get_elem(3, 1), static_cast<T>(11.));
        ASSERT_EQ(test_mat_4_3.get_elem(3, 2), static_cast<T>(12.));

    }

    template <template <typename> typename M, typename T>
    void TestCopyConstructor_Base() {

        M<T> test_mat_2_4(
            *handle_ptr,
            {{static_cast<T>(0.), static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-2.)},
             {static_cast<T>(10.), static_cast<T>(0.), static_cast<T>(12.), static_cast<T>(12.)}}
        );

        M<T> test_mat_copied(test_mat_2_4);
        ASSERT_EQ(test_mat_copied.rows(), 2);
        ASSERT_EQ(test_mat_copied.cols(), 4);
        ASSERT_EQ(test_mat_copied.get_elem(0, 0), static_cast<T>(0.));
        ASSERT_EQ(test_mat_copied.get_elem(0, 1), static_cast<T>(-1.));
        ASSERT_EQ(test_mat_copied.get_elem(0, 2), static_cast<T>(-2.));
        ASSERT_EQ(test_mat_copied.get_elem(0, 3), static_cast<T>(-2.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 0), static_cast<T>(10.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 1), static_cast<T>(0.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 2), static_cast<T>(12.));
        ASSERT_EQ(test_mat_copied.get_elem(1, 3), static_cast<T>(12.));

    }

    template <template <typename> typename M, typename T>
    void TestStaticCreation_Base() {

        constexpr int m_zero(15);
        constexpr int n_zero(17);
        M<T> test_zero(M<T>::Zero(*handle_ptr, m_zero, n_zero));
        ASSERT_EQ(test_zero.rows(), m_zero);
        ASSERT_EQ(test_zero.cols(), n_zero);
        for (int i=0; i<m_zero; ++i) {
            for (int j=0; j<n_zero; ++j) {
                ASSERT_EQ(test_zero.get_elem(i, j), static_cast<T>(0.));
            }
        }

        constexpr int m_one(32);
        constexpr int n_one(13);
        M<T> test_ones(M<T>::Ones(*handle_ptr, m_one, n_one));
        ASSERT_EQ(test_ones.rows(), m_one);
        ASSERT_EQ(test_ones.cols(), n_one);
        for (int i=0; i<m_one; ++i) {
            for (int j=0; j<n_one; ++j) {
                ASSERT_EQ(test_ones.get_elem(i, j), static_cast<T>(1.));
            }
        }

        constexpr int m_identity(40);
        constexpr int n_identity(20);
        M<T> test_identity(M<T>::Identity(*handle_ptr, m_identity, n_identity));
        ASSERT_EQ(test_identity.rows(), m_identity);
        ASSERT_EQ(test_identity.cols(), n_identity);
        for (int i=0; i<m_identity; ++i) {
            for (int j=0; j<n_identity; ++j) {
                if (i == j) { ASSERT_EQ(test_identity.get_elem(i, j), static_cast<T>(1.)); }
                else { ASSERT_EQ(test_identity.get_elem(i, j), static_cast<T>(0.)); }
            }
        }

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from 5 adjacent above and below)
        constexpr int m_rand(40);
        constexpr int n_rand(40);
        M<T> test_rand(M<T>::Random(*handle_ptr, m_rand, n_rand));
        ASSERT_EQ(test_rand.rows(), m_rand);
        ASSERT_EQ(test_rand.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    ((test_rand.get_elem(i, j) != test_rand.get_elem(i-1, j)) ||
                     (test_rand.get_elem(i, j) != test_rand.get_elem(i+1, j)) ||
                     (test_rand.get_elem(i, j) != test_rand.get_elem(i, j-1)) ||
                     (test_rand.get_elem(i, j) != test_rand.get_elem(i, j+1)))
                );
            }
        }

    }

    template <template <typename> typename M, typename T>
    void TestCol_Base() {

        const M<T> const_mat(
            *handle_ptr, 
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

    template <template <typename> typename M>
    void TestBadCol_Base() {

        const int m(4);
        const int n(3);
        M<double> mat(
            *handle_ptr, 
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

    template <template <typename> typename M, typename T>
    void TestScale_Base() {

        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(-8), static_cast<T>(0.8), static_cast<T>(-0.6)},
             {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(-3), static_cast<T>(-2), static_cast<T>(-1)},
             {static_cast<T>(10), static_cast<T>(100), static_cast<T>(1000)}}
        );
        M<T> mat_scaled_mult(mat*static_cast<T>(4));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(mat_scaled_mult.get_elem(i, j),
                          static_cast<T>(4)*mat.get_elem(i, j));
            }
        }
        M<T> mat_scaled_div(mat/static_cast<T>(10));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(mat_scaled_div.get_elem(i, j),
                          (static_cast<T>(1)/static_cast<T>(10))*mat.get_elem(i, j));
            }
        }

    }

    template <template <typename> typename M, typename T>
    void TestMatVec_Base() {

        // Test manually
        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(3), static_cast<T>(2), static_cast<T>(1)}}
        );
        ASSERT_VECTOR_NEAR(
            mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1), static_cast<T>(4), static_cast<T>(7), static_cast<T>(3)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(2), static_cast<T>(5), static_cast<T>(8), static_cast<T>(2)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(3), static_cast<T>(6), static_cast<T>(9), static_cast<T>(1)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1.23), static_cast<T>(4.56), static_cast<T>(7.89), static_cast<T>(3.21)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

        // Test random
        const int m_rand(3);
        const int n_rand(4);
        M<T> rand_mat(M<T>::Random(*handle_ptr, m_rand, n_rand));
        ASSERT_VECTOR_NEAR(
            rand_mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {rand_mat.get_elem(0, 0),
                 rand_mat.get_elem(1, 0),
                 rand_mat.get_elem(2, 0)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {rand_mat.get_elem(0, 1),
                 rand_mat.get_elem(1, 1),
                 rand_mat.get_elem(2, 1)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {rand_mat.get_elem(0, 2),
                 rand_mat.get_elem(1, 2),
                 rand_mat.get_elem(2, 2)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {rand_mat.get_elem(0, 3),
                 rand_mat.get_elem(1, 3),
                 rand_mat.get_elem(2, 3)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01), static_cast<T>(0.001)}
            ),
            MatrixVector<T>(
                *handle_ptr,
                {(static_cast<T>(1)*rand_mat.get_elem(0, 0) +
                  static_cast<T>(0.1)*rand_mat.get_elem(0, 1) +
                  static_cast<T>(0.01)*rand_mat.get_elem(0, 2) +
                  static_cast<T>(0.001)*rand_mat.get_elem(0, 3)),
                 (static_cast<T>(1)*rand_mat.get_elem(1, 0) +
                  static_cast<T>(0.1)*rand_mat.get_elem(1, 1) +
                  static_cast<T>(0.01)*rand_mat.get_elem(1, 2)+
                  static_cast<T>(0.001)*rand_mat.get_elem(1, 3)),
                 (static_cast<T>(1)*rand_mat.get_elem(2, 0) +
                  static_cast<T>(0.1)*rand_mat.get_elem(2, 1) +
                  static_cast<T>(0.01)*rand_mat.get_elem(2, 2)+
                  static_cast<T>(0.001)*rand_mat.get_elem(2, 3))}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

    }

    template <template <typename> typename M, typename T>
    void TestBadMatVec_Base() {

        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );

        MatrixVector<T> vec_too_small(
            *handle_ptr,
            {static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_too_small; }
        );

        MatrixVector<T> vec_too_large(
            *handle_ptr, 
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_too_large; }
        );

        MatrixVector<T> vec_matches_wrong_dimension(
            *handle_ptr,
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_matches_wrong_dimension; }
        );

    }

    template <template <typename> typename M, typename T>
    void TestTransposeMatVec_Base() {

        // Test manually
        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)}}
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(9), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {static_cast<T>(1.59), static_cast<T>(2.61), static_cast<T>(3.72), static_cast<T>(4.83)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

        // Test random
        const int m_rand(3);
        const int n_rand(2);
        M<T> rand_mat(M<T>::Random(*handle_ptr, m_rand, n_rand));
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {rand_mat.get_elem(0, 0),
                 rand_mat.get_elem(0, 1)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {rand_mat.get_elem(1, 0),
                 rand_mat.get_elem(1, 1)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {rand_mat.get_elem(2, 0),
                 rand_mat.get_elem(2, 1)}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                MatrixVector<T>(
                    *handle_ptr,
                    {static_cast<T>(1), static_cast<T>(0.1), static_cast<T>(0.01)}
                )
            ),
            MatrixVector<T>(
                *handle_ptr,
                {(static_cast<T>(1)*rand_mat.get_elem(0, 0) +
                  static_cast<T>(0.1)*rand_mat.get_elem(1, 0) +
                  static_cast<T>(0.01)*rand_mat.get_elem(2, 0)),
                 (static_cast<T>(1)*rand_mat.get_elem(0, 1) +
                  static_cast<T>(0.1)*rand_mat.get_elem(1, 1) +
                  static_cast<T>(0.01)*rand_mat.get_elem(2, 1))}
            ),
            static_cast<T>(2.)*static_cast<T>(Tol<T>::gamma(3))
        );

    }

    template <template <typename> typename M, typename T>
    void TestBadTransposeMatVec_Base() {

        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );

        MatrixVector<T> vec_too_small(
            *handle_ptr,
            {static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_too_small); }
        );

        MatrixVector<T> vec_too_large(
            *handle_ptr,
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_too_large); }
        );

        MatrixVector<T> vec_matches_wrong_dimension(
            *handle_ptr,
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_matches_wrong_dimension); }
        );

    }

    template <template <typename> typename M, typename T>
    void TestTranspose_Base() {

        // Test manually
        constexpr int m_manual(4);
        constexpr int n_manual(3);
        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );
        M<T> mat_transposed(mat.transpose());
        ASSERT_EQ(mat_transposed.rows(), m_manual);
        ASSERT_EQ(mat_transposed.cols(), n_manual);
        for (int i=0; i<m_manual; ++i) {
            for (int j=0; j<n_manual; ++j) {
                ASSERT_EQ(mat_transposed.get_elem(i, j), mat.get_elem(j, i));
            }
        }
        M<T> test(
            *handle_ptr, 
            {{static_cast<T>(1), static_cast<T>(5), static_cast<T>(9)},
             {static_cast<T>(2), static_cast<T>(6), static_cast<T>(10)},
             {static_cast<T>(3), static_cast<T>(7), static_cast<T>(11)},
             {static_cast<T>(4), static_cast<T>(8), static_cast<T>(12)}}
        );
        ASSERT_MATRIX_EQ(mat_transposed, test);

        // Test random
        constexpr int m_rand(12);
        constexpr int n_rand(17);
        M<T> mat_rand(M<T>::Random(*handle_ptr, m_rand, n_rand));
        M<T> mat_rand_transposed(mat_rand.transpose());
        ASSERT_EQ(mat_rand_transposed.rows(), n_rand);
        ASSERT_EQ(mat_rand_transposed.cols(), m_rand);
        for (int i=0; i<n_rand; ++i) {
            for (int j=0; j<m_rand; ++j) {
                ASSERT_EQ(mat_rand_transposed.get_elem(i, j), mat_rand.get_elem(j, i));
            }
        }

    }

    template <template <typename> typename M, typename T>
    void TestMatMat_Base() {

        M<T> mat1(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(4)},
             {static_cast<T>(-1), static_cast<T>(-4)},
             {static_cast<T>(-3), static_cast<T>(-2)},
             {static_cast<T>(10), static_cast<T>(100)}}
        );
        M<T> mat2(
            *handle_ptr,
            {{static_cast<T>(7), static_cast<T>(2), static_cast<T>(-3)},
             {static_cast<T>(10), static_cast<T>(-4), static_cast<T>(-3)}}
        );
        M<T> test_mat(
            *handle_ptr,
            {{static_cast<T>(47), static_cast<T>(-14), static_cast<T>(-15)},
             {static_cast<T>(-47), static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(-41), static_cast<T>(2), static_cast<T>(15)},
             {static_cast<T>(1070), static_cast<T>(-380), static_cast<T>(-330)}}
        );
        ASSERT_MATRIX_EQ(mat1*mat2, test_mat);
        ASSERT_MATRIX_EQ(mat1*M<T>::Identity(*handle_ptr, 2, 2), mat1);
        ASSERT_MATRIX_EQ(mat2*M<T>::Identity(*handle_ptr, 3, 3), mat2);

    }

    template <template <typename> typename M, typename T>
    void TestBadMatMat_Base() {

        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(4)},
             {static_cast<T>(-1), static_cast<T>(-4)},
             {static_cast<T>(-3), static_cast<T>(-2)},
             {static_cast<T>(10), static_cast<T>(100)}}
        );

        M<T> mat_too_small(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat*mat_too_small; });

        M<T> mat_too_big(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat*mat_too_big; });

        M<T> mat_matches_other_dim(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat*mat_matches_other_dim; });

    }

    template <template <typename> typename M, typename T>
    void TestAddSub_Base() {

        M<T> mat1(
            *handle_ptr,
            {{static_cast<T>(47), static_cast<T>(-14), static_cast<T>(-15)},
             {static_cast<T>(-47), static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(-41), static_cast<T>(2), static_cast<T>(15)},
             {static_cast<T>(10), static_cast<T>(-38), static_cast<T>(-33)}}
        );

        M<T> mat2(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(-4), static_cast<T>(-5), static_cast<T>(-6)},
             {static_cast<T>(-7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(10), static_cast<T>(-11), static_cast<T>(-12)}}
        );

        M<T> test_mat_add(
            *handle_ptr,
            {{static_cast<T>(48), static_cast<T>(-12), static_cast<T>(-12)},
             {static_cast<T>(-51), static_cast<T>(9), static_cast<T>(9)},
             {static_cast<T>(-48), static_cast<T>(10), static_cast<T>(24)},
             {static_cast<T>(20), static_cast<T>(-49), static_cast<T>(-45)}}
        );
        ASSERT_MATRIX_EQ(mat1+mat2, test_mat_add);

        M<T> test_mat_sub_1(
            *handle_ptr,
            {{static_cast<T>(46), static_cast<T>(-16), static_cast<T>(-18)},
             {static_cast<T>(-43), static_cast<T>(19), static_cast<T>(21)},
             {static_cast<T>(-34), static_cast<T>(-6), static_cast<T>(6)},
             {static_cast<T>(0), static_cast<T>(-27), static_cast<T>(-21)}}
        );
        ASSERT_MATRIX_EQ(mat1-mat2, test_mat_sub_1);

        M<T> test_mat_sub_2(
            *handle_ptr,
            {{static_cast<T>(-46), static_cast<T>(16), static_cast<T>(18)},
             {static_cast<T>(43), static_cast<T>(-19), static_cast<T>(-21)},
             {static_cast<T>(34), static_cast<T>(6), static_cast<T>(-6)},
             {static_cast<T>(0), static_cast<T>(27), static_cast<T>(21)}}
        );
        ASSERT_MATRIX_EQ(mat2-mat1, test_mat_sub_2);

        ASSERT_MATRIX_EQ(mat1-mat1, M<T>::Zero(*handle_ptr, 4, 3));
        ASSERT_MATRIX_EQ(mat2-mat2, M<T>::Zero(*handle_ptr, 4, 3));

    }

    template <template <typename> typename M, typename T>
    void TestBadAddSub_Base() {

        M<T> mat(
            *handle_ptr,
            {{static_cast<T>(47), static_cast<T>(-14), static_cast<T>(-15)},
             {static_cast<T>(-47), static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(-41), static_cast<T>(2), static_cast<T>(15)},
             {static_cast<T>(10), static_cast<T>(-38), static_cast<T>(-33)}}
        );

        M<T> mat_too_few_cols(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_few_cols; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_few_cols; });

        M<T> mat_too_many_cols(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_many_cols; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_many_cols; });

        M<T> mat_too_few_rows(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_few_rows; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_few_rows; });

        M<T> mat_too_many_rows(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat+mat_too_many_rows; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { mat-mat_too_many_rows; });

    }

    template <template <typename> typename M, typename T>
    void TestNorm_Base() {

        M<T> mat1(
            *handle_ptr,
            {{static_cast<T>(18), static_cast<T>(5)},
             {static_cast<T>(-3), static_cast<T>(6)},
             {static_cast<T>(9), static_cast<T>(-9)},
             {static_cast<T>(4), static_cast<T>(-2)}}
        );
        ASSERT_EQ(mat1.norm(), static_cast<T>(24));

        M<T> mat2(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1)},
             {static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1)},
             {static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(-1)}}
        );
        ASSERT_EQ(mat2.norm(), static_cast<T>(3));

    }

    template <template <typename> typename M>
    void TestCast_Base() {
        
        constexpr int m(20);
        constexpr int n(30);
        M<double> mat_dbl(M<double>::Random(*handle_ptr, m, n));

        M<float> mat_sgl(mat_dbl.template cast<float>());
        ASSERT_EQ(mat_sgl.rows(), m);
        ASSERT_EQ(mat_sgl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(mat_sgl.get_elem(i, j), static_cast<float>(mat_dbl.get_elem(i, j)));
            }
        }

        M<__half> mat_hlf(mat_dbl.template cast<__half>());
        ASSERT_EQ(mat_hlf.rows(), m);
        ASSERT_EQ(mat_hlf.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(mat_hlf.get_elem(i, j), static_cast<__half>(mat_dbl.get_elem(i, j)));
            }
        }

    }

};