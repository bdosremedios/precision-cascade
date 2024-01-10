#ifndef TEST_ASSERTIONS_H
#define TEST_ASSERTIONS_H

#include <functional>

#include "gtest/gtest.h"

#include "types/types.h"

inline void CHECK_FUNC_HAS_RUNTIME_ERROR(std::function<void()> func) {
    try {
        func();
        FAIL();
    } catch (std::runtime_error e) {
        std::cout << e.what() << std::endl;
    }
}

template <typename T>
void ASSERT_VECTOR_NEAR(MatrixVector<T> &test, MatrixVector<T> &target, T tol) {

    ASSERT_EQ(test.rows(), target.rows());

    for (int i=0; i<target.rows(); ++i) {
        ASSERT_NEAR(test.get_elem(i), target.get_elem(i), tol);
    }

}

template <typename T>
void ASSERT_VECTOR_EQ(MatrixVector<T> &test, MatrixVector<T> &target) {
    ASSERT_VECTOR_NEAR(test, target, static_cast<T>(0));
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_NEAR(M<T> test, M<T> target, T tol) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_NEAR(test.get_elem(i, j), target.get_elem(i, j), tol);
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_LT(M<T> test, M<T> target) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_LT(test.coeff(i, j), target.coeff(i, j));
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_GT(M<T> test, M<T> target) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_GT(test.coeff(i, j), target.coeff(i, j));
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_EQ(M<T> test, M<T> target) {
    ASSERT_MATRIX_NEAR(test, target, static_cast<T>(0));
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_SAMESPARSITY(M<T> test, M<T> target, T zero_tol) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            if (abs(target.get_elem(i, j)) <= zero_tol) {
                ASSERT_LE(abs(test.get_elem(i, j)), zero_tol);
            }
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_ZERO(M<T> test, T tol) {
    ASSERT_MATRIX_NEAR(test, M<T>::Zero(test.rows(), test.cols()), tol);
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_IDENTITY(M<T> test, T tol) {
    ASSERT_MATRIX_NEAR(test, M<T>::Identity(test.rows(), test.cols()), tol);
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_LOWTRI(M<T> test, T tol) {

    for (int i=0; i<test.rows(); ++i) {
        for (int j=i+1; j<test.cols(); ++j) {
            ASSERT_NEAR(test.get_elem(i, j), static_cast<T>(0), tol);
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_UPPTRI(M<T> test, T tol) {

    for (int i=0; i<test.rows(); ++i) {
        for (int j=0; j<i; ++j) {
            ASSERT_NEAR(test.get_elem(i, j), static_cast<T>(0), tol);
        }
    }

}

#endif