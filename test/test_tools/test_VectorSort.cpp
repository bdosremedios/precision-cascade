#include "../test.h"

#include "tools/VectorSort.h"

class VectorSort_Test: public TestBase
{
public:

    template <typename T>
    void simple_sort_even() {

        constexpr int n(10);
        MatrixVector<T> vec(n);
        MatrixVector<int> target(n);
        vec(0) = static_cast<T>(3.519); vec(1) = static_cast<T>(8.525);
        vec(2) = static_cast<T>(3.978); vec(3) = static_cast<T>(8.645);
        vec(4) = static_cast<T>(2.798); vec(5) = static_cast<T>(1.477);
        vec(6) = static_cast<T>(7.021); vec(7) = static_cast<T>(5.689);
        vec(8) = static_cast<T>(6.185); vec(9) = static_cast<T>(6.315);
        target(0) = 5; target(1) = 4; target(2) = 0; target(3) = 2;
        target(4) = 7; target(5) = 8; target(6) = 9; target(7) = 6;
        target(8) = 1; target(9) = 3;

        MatrixVector<int> test = sort_indices(vec);

        ASSERT_VECTOR_EQ(target, test);

    }

    template <typename T>
    void simple_sort_odd() {

        constexpr int n(13);
        MatrixVector<T> vec(n);
        MatrixVector<int> target(n);
        vec(0) = static_cast<T>(7.063); vec(1) = static_cast<T>(8.824); vec(2) = static_cast<T>(5.430);
        vec(3) = static_cast<T>(5.107); vec(4) = static_cast<T>(5.478); vec(5) = static_cast<T>(8.819);
        vec(6) = static_cast<T>(7.995); vec(7) = static_cast<T>(9.787); vec(8) = static_cast<T>(4.139);
        vec(9) = static_cast<T>(8.946); vec(10) = static_cast<T>(4.861); vec(11) = static_cast<T>(1.678);
        vec(12) = static_cast<T>(9.176);
        target(0) = 11; target(1) = 8; target(2) = 10;
        target(3) = 3; target(4) = 2; target(5) = 4;
        target(6) = 0; target(7) = 6; target(8) = 5;
        target(9) = 1; target(10) = 9; target(11) = 12;
        target(12) = 7;

        MatrixVector<int> test = sort_indices(vec);

        ASSERT_VECTOR_EQ(target, test);

    }

    template <typename T>
    void simple_sort_duplicates() {

        constexpr int n(7);
        MatrixVector<T> vec(n);
        MatrixVector<int> target(n);
        vec(0) = static_cast<T>(9.433); vec(1) = static_cast<T>(3.950); vec(2) = static_cast<T>(1.776);
        vec(3) = static_cast<T>(7.016); vec(4) = static_cast<T>(1.409); vec(5) = static_cast<T>(1.776);
        vec(6) = static_cast<T>(7.016);
        target(0) = 4; target(1) = 2; target(2) = 5;
        target(3) = 1; target(4) = 3; target(5) = 6;
        target(6) = 0;

        MatrixVector<int> test = sort_indices(vec);

        // Check that sorted element order is the same for all elements
        for (int i=0; i<n; ++i) {
            ASSERT_EQ(vec(target(i)), vec(test(i)));
        }

    }

    template <typename T>
    void sorted_already() {

        constexpr int n(10);
        MatrixVector<T> vec(n);
        MatrixVector<int> target(n);
        vec(0) = static_cast<T>(1); vec(1) = static_cast<T>(2);
        vec(2) = static_cast<T>(3); vec(3) = static_cast<T>(4);
        vec(4) = static_cast<T>(5); vec(5) = static_cast<T>(6);
        vec(6) = static_cast<T>(7); vec(7) = static_cast<T>(8);
        vec(8) = static_cast<T>(9); vec(9) = static_cast<T>(10);
        target(0) = 0; target(1) = 1; target(2) = 2; target(3) = 3;
        target(4) = 4; target(5) = 5; target(6) = 6; target(7) = 7;
        target(8) = 8; target(9) = 9;

        MatrixVector<int> test = sort_indices(vec);

        ASSERT_VECTOR_EQ(target, test);

    }

    template <typename T>
    void one_element() {

        constexpr int n(1);
        MatrixVector<T> vec(n);
        MatrixVector<int> target(n);
        vec(0) = static_cast<T>(1);
        target(0) = 0;

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