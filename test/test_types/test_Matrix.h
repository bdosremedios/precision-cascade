#include "../test.h"

class Matrix_Test: public TestBase
{
protected:

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

};