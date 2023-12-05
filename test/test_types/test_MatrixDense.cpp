#include "../test.h"

#include "types/MatrixDense.h"

class MatrixDense_Test: public TestBase
{
public:

    template <typename T>
    void TestListInitialization() {

        MatrixDense<T> test_mat_0_0 ({});
        ASSERT_EQ(test_mat_0_0.rows(), 0);
        ASSERT_EQ(test_mat_0_0.cols(), 0);

        MatrixDense<T> test_mat_0_1 ({{}});
        ASSERT_EQ(test_mat_0_1.rows(), 1);
        ASSERT_EQ(test_mat_0_1.cols(), 0);

        MatrixDense<T> test_mat_1 (
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

        MatrixDense<T> test_mat_2 (
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
        
        MatrixDense<T> test_mat_3 (
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

};

TEST_F(MatrixDense_Test, TestListInitialization_Hlf) { TestListInitialization<half>(); }
TEST_F(MatrixDense_Test, TestListInitialization_Sgl) { TestListInitialization<float>(); }
TEST_F(MatrixDense_Test, TestListInitialization_Dbl) { TestListInitialization<double>(); }