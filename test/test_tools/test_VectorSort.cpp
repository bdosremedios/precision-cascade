#include "../test.h"

#include "tools/VectorSort.h"

class VectorSort_Test: public TestBase
{
public:

    template <typename T>
    void simple_sort_even() {

        MatrixVector<T> vec ({
            static_cast<T>(3.519), static_cast<T>(8.525), static_cast<T>(3.978), static_cast<T>(8.645),
            static_cast<T>(2.798), static_cast<T>(1.477), static_cast<T>(7.021), static_cast<T>(5.689),
            static_cast<T>(6.185), static_cast<T>(6.315)
        });
        MatrixVector<int> target({5, 4, 0, 2, 7, 8, 9, 6, 1, 3});

        MatrixVector<int> test = sort_indices(vec);

        ASSERT_VECTOR_EQ(target, test);

    }

    template <typename T>
    void simple_sort_odd() {

        MatrixVector<T> vec ({
            static_cast<T>(7.063), static_cast<T>(8.824), static_cast<T>(5.430), static_cast<T>(5.107),
            static_cast<T>(5.478), static_cast<T>(8.819), static_cast<T>(7.995), static_cast<T>(9.787),
            static_cast<T>(4.139), static_cast<T>(8.946), static_cast<T>(4.861), static_cast<T>(1.678),
            static_cast<T>(9.176)
        });
        MatrixVector<int> target({11, 8, 10, 3, 2, 4, 0, 6, 5, 1, 9, 12, 7});

        MatrixVector<int> test = sort_indices(vec);
        ASSERT_VECTOR_EQ(target, test);

    }

    template <typename T>
    void simple_sort_duplicates() {

        constexpr int n(7);
        MatrixVector<T> vec ({
            static_cast<T>(9.433), static_cast<T>(3.950), static_cast<T>(1.776), static_cast<T>(7.016),
            static_cast<T>(1.409), static_cast<T>(1.776), static_cast<T>(7.016)
        });
        MatrixVector<int> target({4, 2, 5, 1, 3, 6, 0});

        MatrixVector<int> test = sort_indices(vec);

        // Check that sorted element order is the same for all elements
        for (int i=0; i<n; ++i) { ASSERT_EQ(vec(target(i)), vec(test(i))); }

    }

    template <typename T>
    void sorted_already() {

        MatrixVector<T> vec ({
            static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4),
            static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8),
            static_cast<T>(9), static_cast<T>(10)
        });
        MatrixVector<int> target({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        MatrixVector<int> test = sort_indices(vec);

        ASSERT_VECTOR_EQ(target, test);

    }

    template <typename T>
    void one_element() {

        MatrixVector<T> vec({static_cast<T>(1)});
        MatrixVector<int> target({0});

        MatrixVector<int> test = sort_indices(vec);

        ASSERT_VECTOR_EQ(target, test);

    }

};

TEST_F(VectorSort_Test, TestSimpleSortHalf_EvenNum) { simple_sort_even<half>(); }
TEST_F(VectorSort_Test, TestSimpleSortSingle_EvenNum) { simple_sort_even<float>(); }
TEST_F(VectorSort_Test, TestSimpleSortDouble_EvenNum) { simple_sort_even<double>(); }

TEST_F(VectorSort_Test, TestSimpleSortHalf_OddNum) { simple_sort_odd<half>(); }
TEST_F(VectorSort_Test, TestSimpleSortSingle_OddNum) { simple_sort_odd<float>(); }
TEST_F(VectorSort_Test, TestSimpleSortDouble_OddNum) { simple_sort_odd<double>(); }

TEST_F(VectorSort_Test, TestSimpleSortHalf_Dupes) { simple_sort_duplicates<half>(); }
TEST_F(VectorSort_Test, TestSimpleSortSingle_Dupes) { simple_sort_duplicates<float>(); }
TEST_F(VectorSort_Test, TestSimpleSortDouble_Dupes) { simple_sort_duplicates<double>(); }

TEST_F(VectorSort_Test, TestAlreadySorted) { sorted_already<double>(); }

TEST_F(VectorSort_Test, TestOneElement) { one_element<double>(); }