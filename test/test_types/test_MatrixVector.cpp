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

    }

    template <typename T>
    void TestConstruction() {
        
        

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

};

TEST_F(MatrixVector_Test, TestElementAccessMethods_Hlf) { TestElementAccessMethods<half>(); }
TEST_F(MatrixVector_Test, TestElementAccessMethods_Sgl) { TestElementAccessMethods<float>(); }
TEST_F(MatrixVector_Test, TestElementAccessMethods_Dbl) { TestElementAccessMethods<double>(); }

TEST_F(MatrixVector_Test, TestPropertyMethods_Hlf) { TestPropertyMethods<half>(); }
TEST_F(MatrixVector_Test, TestPropertyMethods_Sgl) { TestPropertyMethods<float>(); }
TEST_F(MatrixVector_Test, TestPropertyMethods_Dbl) { TestPropertyMethods<double>(); }