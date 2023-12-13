#include "../test.h"

class Matrix_Test: public TestBase
{
protected:

    template <template <typename> typename M, typename T>
    void TestConstruction_Base() {

        M<T> test_mat_empty;
        ASSERT_EQ(test_mat_empty.rows(), 0);
        ASSERT_EQ(test_mat_empty.cols(), 0);

        constexpr int m(12);
        M<T> test_mat_square(m, m);
        ASSERT_EQ(test_mat_square.rows(), m);
        ASSERT_EQ(test_mat_square.cols(), m);

        constexpr int n(33);
        M<T> test_mat_wide(m, n);
        ASSERT_EQ(test_mat_wide.rows(), m);
        ASSERT_EQ(test_mat_wide.cols(), n);

        M<T> test_mat_tall(n, m);
        ASSERT_EQ(test_mat_tall.rows(), n);
        ASSERT_EQ(test_mat_tall.cols(), m);

    }

    template <template <typename> typename M, typename T>
    void TestCoeffAccess_Base() {
        
        constexpr int m(24);
        constexpr int n(12);
        M<T> test_mat(m, n);

        T elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                test_mat.coeffRef(i, j) = elem;
                elem += static_cast<T>(1);
            }
        }

        T test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat.coeff(i, j), test_elem);
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 0 row as all -1
        for (int j=0; j<n; ++j) { test_mat.coeffRef(0, j) = static_cast<T>(-1.); }
        test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) { ASSERT_EQ(test_mat.coeff(i, j), static_cast<T>(-1.)); }
                else { ASSERT_EQ(test_mat.coeff(i, j), test_elem); }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 4 row as decreasing by -1 from -1
        T row_5_elem = static_cast<T>(-1.);
        for (int j=0; j<n; ++j) { test_mat.coeffRef(4, j) = row_5_elem; row_5_elem += static_cast<T>(-1.);}
        test_elem = static_cast<T>(1);
        T row_5_test_elem = static_cast<T>(-1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) { ASSERT_EQ(test_mat.coeff(i, j), static_cast<T>(-1.)); }
                else if (i == 4) { ASSERT_EQ(test_mat.coeff(i, j), row_5_test_elem);
                                   row_5_test_elem += static_cast<T>(-1.);}
                else { ASSERT_EQ(test_mat.coeff(i, j), test_elem); }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 2 col as incresing by 1 from -5
        T coL_3_elem = static_cast<T>(-5.);
        for (int i=0; i<m; ++i) { test_mat.coeffRef(i, 2) = coL_3_elem; coL_3_elem += static_cast<T>(1.);}
        test_elem = static_cast<T>(1);
        row_5_test_elem = static_cast<T>(-1.);
        T coL_3_test_elem = static_cast<T>(-5.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (j == 2) { ASSERT_EQ(test_mat.coeff(i, j), coL_3_test_elem); coL_3_test_elem += static_cast<T>(1.);}
                else if (i == 0) { ASSERT_EQ(test_mat.coeff(i, j), static_cast<T>(-1.)); }
                else if (i == 4) { ASSERT_EQ(test_mat.coeff(i, j), row_5_test_elem);
                                   row_5_test_elem += static_cast<T>(-1.);
                                   if (j == 1) { row_5_test_elem += static_cast<T>(-1.); } }
                else { ASSERT_EQ(test_mat.coeff(i, j), test_elem); }
                test_elem += static_cast<T>(1);
            }
        }

    }

    template <template <typename> typename M, typename T>
    void TestPropertyAccess_Base();

    template <template <typename> typename M, typename T>
    void TestListInitialization_Base() {

        M<T> test_mat_0_0 ({});
        ASSERT_EQ(test_mat_0_0.rows(), 0);
        ASSERT_EQ(test_mat_0_0.cols(), 0);

        M<T> test_mat_0_1 ({{}});
        ASSERT_EQ(test_mat_0_1.rows(), 1);
        ASSERT_EQ(test_mat_0_1.cols(), 0);

        M<T> test_mat_1 (
            {{static_cast<T>(5.), static_cast<T>(3.), static_cast<T>(27.)},
             {static_cast<T>(88.), static_cast<T>(-4.), static_cast<T>(-6.)},
             {static_cast<T>(100.), static_cast<T>(12.), static_cast<T>(2.)}}
        );
        ASSERT_EQ(test_mat_1.rows(), 3);
        ASSERT_EQ(test_mat_1.cols(), 3);
        ASSERT_EQ(test_mat_1.coeff(0, 0), static_cast<T>(5.));
        ASSERT_EQ(test_mat_1.coeff(0, 1), static_cast<T>(3.));
        ASSERT_EQ(test_mat_1.coeff(0, 2), static_cast<T>(27.));
        ASSERT_EQ(test_mat_1.coeff(1, 0), static_cast<T>(88.));
        ASSERT_EQ(test_mat_1.coeff(1, 1), static_cast<T>(-4.));
        ASSERT_EQ(test_mat_1.coeff(1, 2), static_cast<T>(-6.));
        ASSERT_EQ(test_mat_1.coeff(2, 0), static_cast<T>(100.));
        ASSERT_EQ(test_mat_1.coeff(2, 1), static_cast<T>(12.));
        ASSERT_EQ(test_mat_1.coeff(2, 2), static_cast<T>(2.));

        M<T> test_mat_2 (
            {{static_cast<T>(7.), static_cast<T>(5.), static_cast<T>(3.)},
             {static_cast<T>(1.), static_cast<T>(6.), static_cast<T>(2.)}}
        );
        ASSERT_EQ(test_mat_2.rows(), 2);
        ASSERT_EQ(test_mat_2.cols(), 3);
        ASSERT_EQ(test_mat_2.coeff(0, 0), static_cast<T>(7.));
        ASSERT_EQ(test_mat_2.coeff(0, 1), static_cast<T>(5.));
        ASSERT_EQ(test_mat_2.coeff(0, 2), static_cast<T>(3.));
        ASSERT_EQ(test_mat_2.coeff(1, 0), static_cast<T>(1.));
        ASSERT_EQ(test_mat_2.coeff(1, 1), static_cast<T>(6.));
        ASSERT_EQ(test_mat_2.coeff(1, 2), static_cast<T>(2.));
        
        M<T> test_mat_3 (
            {{static_cast<T>(7.), static_cast<T>(5.)},
             {static_cast<T>(1.), static_cast<T>(6.)},
             {static_cast<T>(3.), static_cast<T>(2.)},
             {static_cast<T>(43.), static_cast<T>(9.)}}
        );
        ASSERT_EQ(test_mat_3.rows(), 4);
        ASSERT_EQ(test_mat_3.cols(), 2);
        ASSERT_EQ(test_mat_3.coeff(0, 0), static_cast<T>(7.));
        ASSERT_EQ(test_mat_3.coeff(0, 1), static_cast<T>(5.));
        ASSERT_EQ(test_mat_3.coeff(1, 0), static_cast<T>(1.));
        ASSERT_EQ(test_mat_3.coeff(1, 1), static_cast<T>(6.));
        ASSERT_EQ(test_mat_3.coeff(2, 0), static_cast<T>(3.));
        ASSERT_EQ(test_mat_3.coeff(2, 1), static_cast<T>(2.));
        ASSERT_EQ(test_mat_3.coeff(3, 0), static_cast<T>(43.));
        ASSERT_EQ(test_mat_3.coeff(3, 1), static_cast<T>(9.));

    }

    template <template <typename> typename M>
    void TestBadListInitialization_Base() {

        try {
            M<double> mat ({
                {1, 2, 3, 4},
                {1, 2, 3}
            });
            FAIL();
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
        }

        try {
            M<double> mat ({
                {1, 2},
                {1, 2, 3}
            });
            FAIL();
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
        }

        try {
            M<double> mat ({
                {1, 2},
                {}
            });
            FAIL();
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
        }

        try {
            M<double> mat ({
                {},
                {1, 2, 3}
            });
            FAIL();
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
        }

    }

    template <template <typename> typename M, typename T>
    void TestStaticCreation_Base() {

        constexpr int m_zero(15);
        constexpr int n_zero(17);
        M<T> test_zero(M<T>::Zero(m_zero, n_zero));
        ASSERT_EQ(test_zero.rows(), m_zero);
        ASSERT_EQ(test_zero.cols(), n_zero);
        for (int i=0; i<m_zero; ++i) {
            for (int j=0; j<n_zero; ++j) {
                ASSERT_EQ(test_zero.coeff(i, j), static_cast<T>(0.));
            }
        }

        constexpr int m_one(32);
        constexpr int n_one(13);
        M<T> test_ones(M<T>::Ones(m_one, n_one));
        ASSERT_EQ(test_ones.rows(), m_one);
        ASSERT_EQ(test_ones.cols(), n_one);
        for (int i=0; i<m_one; ++i) {
            for (int j=0; j<n_one; ++j) {
                ASSERT_EQ(test_ones.coeff(i, j), static_cast<T>(1.));
            }
        }

        constexpr int m_identity(40);
        constexpr int n_identity(20);
        M<T> test_identity(M<T>::Identity(m_identity, n_identity));
        ASSERT_EQ(test_identity.rows(), m_identity);
        ASSERT_EQ(test_identity.cols(), n_identity);
        for (int i=0; i<m_identity; ++i) {
            for (int j=0; j<n_identity; ++j) {
                if (i == j) { ASSERT_EQ(test_identity.coeff(i, j), static_cast<T>(1.)); }
                else { ASSERT_EQ(test_identity.coeff(i, j), static_cast<T>(0.)); }
            }
        }

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from 5 adjacent above and below)
        constexpr int m_rand(40);
        constexpr int n_rand(40);
        M<T> test_rand(M<T>::Random(m_rand, n_rand));
        ASSERT_EQ(test_rand.rows(), m_rand);
        ASSERT_EQ(test_rand.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    ((test_rand.coeff(i, j) != test_rand.coeff(i-1, j)) ||
                     (test_rand.coeff(i, j) != test_rand.coeff(i+1, j)) ||
                     (test_rand.coeff(i, j) != test_rand.coeff(i, j-1)) ||
                     (test_rand.coeff(i, j) != test_rand.coeff(i, j+1)))
                );
            }
        }

    }

    template <template <typename> typename M, typename T>
    void TestCol_Base() {

        const M<T> const_mat ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
            {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
            {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
            {static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}
        });
        M<T> mat(const_mat);
        
        // Test cast/access
        for (int j=0; j<3; ++j) {
            MatrixVector<T> vec(mat.col(j));
            for (int i=0; i<4; ++i) {
                ASSERT_EQ(vec(i), const_mat.coeff(i, j));
            }
        }

        // Test norm
        MatrixVector<T> norm_vec(mat.col(1));
        ASSERT_EQ(mat.col(1).norm(), norm_vec.norm());

        // Test assignment
        MatrixVector<T> assign_vec({static_cast<T>(1),
                                    static_cast<T>(1),
                                    static_cast<T>(1),
                                    static_cast<T>(1)});
        mat.col(2) = assign_vec;
        for (int j=0; j<2; ++j) {
            for (int i=0; i<4; ++i) {
                ASSERT_EQ(mat.coeff(i, j), const_mat.coeff(i, j));
            }
        }
        for (int i=0; i<4; ++i) { ASSERT_EQ(mat.coeff(i, 2), static_cast<T>(1)); }

    }

    template <template <typename> typename M, typename T>
    void TestBlock_Base()  {

        const M<T> const_mat ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
            {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(10)},
            {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14), static_cast<T>(15)},
            {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(20)}
        });
        M<T> mat(const_mat);
        
        // Test cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4(mat.block(0, 0, 3, 4));
        MatrixDense<T> test_0_0_3_4 ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
            {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
            {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14)}
        });
        ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

        // Test cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1(mat.block(1, 2, 3, 1));
        MatrixDense<T> test_1_2_3_1 ({
            {static_cast<T>(8)},
            {static_cast<T>(13)},
            {static_cast<T>(18)}
        });
        ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

    }

    template <template <typename> typename M, typename T>
    void TestTranspose_Base();

    template <template <typename> typename M, typename T>
    void TestScale_Base();

    template <template <typename> typename M, typename T>
    void TestMatVec_Base();

    template <template <typename> typename M, typename T>
    void TestMatMat_Base();

    template <template <typename> typename M, typename T>
    void TestNorm_Base();

    template <template <typename> typename M, typename T>
    void TestAddSub_Base();

    template <template <typename> typename M, typename T>
    void TestCast_Base();

};