#include "../test.h"

#include <vector>
#include <random>
#include <algorithm>

#include "tools/Sort.h"

class Sort_Test: public TestBase
{
public:

    template <typename T, typename W>
    void TestManualSort() {

        constexpr int n(9);

        T arr_1[n] = { static_cast<T>(2.), static_cast<T>(1.), static_cast<T>(9.),
                       static_cast<T>(4.), static_cast<T>(7.), static_cast<T>(8.),
                       static_cast<T>(5.), static_cast<T>(3.), static_cast<T>(6.) };
        W arr_2[n] = { static_cast<W>(1.), static_cast<W>(2.), static_cast<W>(3.),
                       static_cast<W>(4.), static_cast<W>(5.), static_cast<W>(6.),
                       static_cast<W>(7.), static_cast<W>(8.), static_cast<W>(9.) };

        sort::in_place_passengered_sort(
            0, 9, arr_1, arr_2
        );

        T arr_1_soln[n] = { static_cast<T>(1.), static_cast<T>(2.), static_cast<T>(3.),
                            static_cast<T>(4.), static_cast<T>(5.), static_cast<T>(6.),
                            static_cast<T>(7.), static_cast<T>(8.), static_cast<T>(9.) };
        W arr_2_soln[n] = { static_cast<W>(2.), static_cast<W>(1.), static_cast<W>(8.),
                            static_cast<W>(4.), static_cast<W>(7.), static_cast<W>(9.),
                            static_cast<W>(5.), static_cast<W>(6.), static_cast<W>(3.) };

        for (int k=0; k<n; ++k) {
            ASSERT_EQ(arr_1[k], arr_1_soln[k]);
            ASSERT_EQ(arr_2[k], arr_2_soln[k]);
        }

    }

    template <typename T, typename W>
    void TestRandomSort() {

        srand(time(NULL));
        const int n(10 + (rand() % 100));

        std::vector<T> arr_1;
        for (int k=0; k<n; ++k) { arr_1.push_back(static_cast<T>(k)); }
        std::vector<W> arr_2;
        for (int k=0; k<n; ++k) { arr_2.push_back(static_cast<W>(k)); }

        std::shuffle(arr_1.begin(), arr_1.end(), std::default_random_engine(rand()));

        std::vector<T> arr_1_soln;
        for (int k=0; k<n; ++k) { arr_1_soln.push_back(static_cast<T>(k)); }
        std::vector<W> arr_2_soln;
        arr_2_soln.resize(n);
        for (int k=0; k<n; ++k) { arr_2_soln[static_cast<int>(arr_1[k])] = static_cast<W>(k); }

        sort::in_place_passengered_sort(0, n, &arr_1[0], &arr_2[0]);

        for (int k=0; k<n; ++k) {
            ASSERT_EQ(arr_1[k], arr_1_soln[k]);
            ASSERT_EQ(arr_2[k], arr_2_soln[k]);
        }

    }

};

TEST_F(Sort_Test, TestMatchingIntSort) {
    TestManualSort<int, int>();
}

TEST_F(Sort_Test, TestMatchingDoubleSort) {
    TestManualSort<double, double>();
}

TEST_F(Sort_Test, TestIntDoubleSort) {
    TestManualSort<int, double>();
}

TEST_F(Sort_Test, TestDoubleIntSort) {
    TestManualSort<double, int>();
}

TEST_F(Sort_Test, TestRandomMatchingIntSort) {
    TestRandomSort<int, int>();
}

TEST_F(Sort_Test, TestRandomMatchingDoubleSort) {
    TestRandomSort<double, double>();
}

TEST_F(Sort_Test, TestRandomIntDoubleSort) {
    TestRandomSort<int, double>();
}

TEST_F(Sort_Test, TestRandomDoubleIntSort) {
    TestRandomSort<double, int>();
}