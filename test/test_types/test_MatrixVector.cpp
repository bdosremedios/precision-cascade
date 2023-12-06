#include "../test.h"

#include "types/MatrixVector.h"


class MatrixVector_Test: public TestBase
{
public:

    template <typename T>
    void TestElementAccessMethods() {
        
        constexpr int n(10);
        MatrixVector<T> test_vec_n(n);
        for (int i=0; i<n; ++i) { test_vec_n(i) = static_cast<T>(i*i); }
        for (int i=0; i<n; ++i) { ASSERT_EQ(test_vec_n(i), static_cast<T>(i*i)); }
        
        constexpr int m(18);
        MatrixVector<T> test_vec_m(m);
        for (int i=0; i<n; ++i) { test_vec_m(i) = static_cast<T>(2*i*i-m); }
        for (int i=0; i<n; ++i) { ASSERT_EQ(test_vec_m(i), static_cast<T>(2*i*i-m)); }
        
        constexpr int cnt(10);
        MatrixVector<T> test_vec_cnt(cnt);
        for (int i=0; i<n; ++i) { test_vec_cnt(i) = static_cast<T>(1+i); }
        for (int i=0; i<n; ++i) { ASSERT_EQ(test_vec_cnt(i), static_cast<T>(1+i)); }

        MatrixVector<T> test_vec_cnt_2_6(test_vec_cnt.slice(2, 6));
        ASSERT_EQ(test_vec_cnt_2_6.rows(), 6);
        for (int i=0; i<6; ++i) { ASSERT_EQ(test_vec_cnt_2_6(i), test_vec_cnt(i+2)); }
        
        MatrixVector<T> test_vec_m_1_3(test_vec_m.slice(1, 3));
        ASSERT_EQ(test_vec_m_1_3.rows(), 3);
        for (int i=0; i<3; ++i) { ASSERT_EQ(test_vec_m_1_3(i), test_vec_m(i+1)); }

        MatrixVector<T> test_vec_n_0_4(test_vec_n.slice(0, 4));
        ASSERT_EQ(test_vec_n_0_4.rows(), 4);
        for (int i=0; i<4; ++i) { ASSERT_EQ(test_vec_n_0_4(i), test_vec_n(i)); }

        MatrixVector<T> test_vec_m_dupe(test_vec_m.slice(0, m));
        ASSERT_EQ(test_vec_m_dupe.rows(), m);
        ASSERT_VECTOR_EQ(test_vec_m_dupe, test_vec_m);

    }

    template <typename T>
    void TestPropertyMethods() {
        
        constexpr int m(8);
        MatrixVector<T> test_vec_m(m);

        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);
        
        constexpr int n(16);
        MatrixVector<T> test_vec_n(n);
        for (int i=0; i<n; ++i) { test_vec_n(i) = static_cast<T>(i*i); }

        ASSERT_EQ(test_vec_n.rows(), n);
        ASSERT_EQ(test_vec_n.cols(), 1);

    } 

    void TestConstruction() {

        MatrixVector<double> test_vec_0(0);
        ASSERT_EQ(test_vec_0.rows(), 0);
        ASSERT_EQ(test_vec_0.cols(), 1);

        constexpr int m(24);
        MatrixVector<double> test_vec_m(m);
        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);

    }

    template <typename T>
    void TestListInitialization() {
        
        MatrixVector<T> test_vec({static_cast<T>(7.), static_cast<T>(5.), static_cast<T>(3.),
                                  static_cast<T>(1.), static_cast<T>(6.), static_cast<T>(2.)});
        ASSERT_EQ(test_vec(0), static_cast<T>(7.));
        ASSERT_EQ(test_vec(1), static_cast<T>(5.));
        ASSERT_EQ(test_vec(2), static_cast<T>(3.));
        ASSERT_EQ(test_vec(3), static_cast<T>(1.));
        ASSERT_EQ(test_vec(4), static_cast<T>(6.));
        ASSERT_EQ(test_vec(5), static_cast<T>(2.));

    }

    template <typename T>
    void TestStaticCreation() {

        constexpr int m_zero(15);
        MatrixVector<T> test_zero(MatrixVector<T>::Zero(m_zero));
        ASSERT_EQ(test_zero.rows(), m_zero);
        for (int i=0; i<m_zero; ++i) { ASSERT_EQ(test_zero(i), static_cast<T>(0.)); }

        constexpr int m_one(15);
        MatrixVector<T> test_ones(MatrixVector<T>::Ones(m_one));
        ASSERT_EQ(test_ones.rows(), m_one);
        for (int i=0; i<m_one; ++i) { ASSERT_EQ(test_ones(i), static_cast<T>(1.)); }

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from other 5 (so only will fail if some 5 numbers in a row are exactly
        // the same))
        constexpr int m_rand(200);
        MatrixVector<T> test_rand(MatrixVector<T>::Random(m_rand));
        ASSERT_EQ(test_rand.rows(), m_rand);
        for (int i=2; i<m_one-2; ++i) {
            ASSERT_TRUE(
                ((test_rand(i) != test_rand(i-2)) ||
                 (test_rand(i) != test_rand(i-1)) ||
                 (test_rand(i) != test_rand(i+1)) ||
                 (test_rand(i) != test_rand(i+2)))
            );
        }

    }

    template <typename T>
    void TestAssignment() {

        constexpr int m(23);
        constexpr int n(16);
        MatrixVector<T> test_vec(MatrixVector<T>::Zero(m));
        ASSERT_EQ(test_vec.rows(), m);
        ASSERT_VECTOR_EQ(test_vec, MatrixVector<T>::Zero(m));

        test_vec = MatrixVector<T>::Ones(n);
        ASSERT_EQ(test_vec.rows(), n);
        ASSERT_VECTOR_EQ(test_vec, MatrixVector<T>::Ones(n));

    }

    template <typename T>
    void TestDot() {

        // Test Dot
        MatrixVector<T> vec_1_dot({
            static_cast<T>(-4.), static_cast<T>(3.4), static_cast<T>(0.),
            static_cast<T>(-2.1), static_cast<T>(1.8)
        });
        MatrixVector<T> vec_2_dot({
            static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(1.5),
            static_cast<T>(-4.5), static_cast<T>(2.)
        });
        ASSERT_NEAR(vec_1_dot.dot(vec_2_dot), static_cast<T>(11.05), static_cast<T>(11.05)*Tol<T>::gamma(5));

    }

    template <typename T>
    void TestNorm();

    void TestCast();

};

TEST_F(MatrixVector_Test, TestElementAccessMethods) {
    TestElementAccessMethods<half>();
    TestElementAccessMethods<float>();
    TestElementAccessMethods<double>();
}

TEST_F(MatrixVector_Test, TestPropertyMethods) {
    TestPropertyMethods<half>();
    TestPropertyMethods<float>();
    TestPropertyMethods<double>();
}

TEST_F(MatrixVector_Test, TestConstruction) { TestConstruction(); }

TEST_F(MatrixVector_Test, TestListInitialization) {
    TestListInitialization<half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixVector_Test, TestStaticCreation) {
    TestStaticCreation<half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(MatrixVector_Test, TestAssignment) {
    TestAssignment<half>();
    TestAssignment<float>();
    TestAssignment<double>();
}

TEST_F(MatrixVector_Test, TestDot) { TestDot<half>(); TestDot<float>(); TestDot<double>(); }