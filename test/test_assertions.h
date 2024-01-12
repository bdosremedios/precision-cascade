#ifndef TEST_ASSERTIONS_H
#define TEST_ASSERTIONS_H

#include <functional>
#include <string>

#include "gtest/gtest.h"

#include "types/types.h"

static bool *print_errors;

#define CHECK_FUNC_HAS_RUNTIME_ERROR(to_print, func) \
_CHECK_FUNC_HAS_RUNTIME_ERROR( \
    to_print, func, __FILE__, __LINE__, "AssertionError for CHECK_FUNC_HAS_RUNTIME_ERROR" \
)

inline void _CHECK_FUNC_HAS_RUNTIME_ERROR(
    bool *to_print, std::function<void()> func,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    try {
        func();
        FAIL();
    } catch (std::runtime_error e) {
        if (*to_print) {
            std::cout << e.what() << std::endl;
        }
    }

}

#define ASSERT_VECTOR_NEAR(test, target, tol) \
_ASSERT_VECTOR_NEAR( \
    test, target, tol, __FILE__, __LINE__, "AssertionError for ASSERT_VECTOR_NEAR" \
)

template <typename T>
void _ASSERT_VECTOR_NEAR(
    MatrixVector<T> &test, MatrixVector<T> &target, T tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());

    for (int i=0; i<target.rows(); ++i) {
        ASSERT_NEAR(test.get_elem(i), target.get_elem(i), tol);
    }

}

#define ASSERT_VECTOR_EQ(test, target) \
_ASSERT_VECTOR_EQ( \
    test, target, __FILE__, __LINE__, "AssertionError for ASSERT_VECTOR_EQ" \
)

template <typename T>
void _ASSERT_VECTOR_EQ(
    MatrixVector<T> &test, MatrixVector<T> &target,
    const char* file, int line, std::string message
) {
    _ASSERT_VECTOR_NEAR(test, target, static_cast<T>(0), file, line, message);
}

#define ASSERT_MATRIX_NEAR(test, target, tol) \
_ASSERT_MATRIX_NEAR( \
    test, target, tol, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_NEAR" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_NEAR(
    M<T> test, M<T> target, T tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_NEAR(test.get_elem(i, j), target.get_elem(i, j), tol);
        }
    }

}

#define ASSERT_MATRIX_LT(test, target) \
_ASSERT_MATRIX_LT( \
    test, target, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_LT" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_LT(
    M<T> test, M<T> target,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_LT(test.get_elem(i, j), target.get_elem(i, j));
        }
    }

}

#define ASSERT_MATRIX_GT(test, target) \
_ASSERT_MATRIX_GT( \
    test, target, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_GT" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_GT(
    M<T> test, M<T> target,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_GT(test.get_elem(i, j), target.get_elem(i, j));
        }
    }

}

#define ASSERT_MATRIX_EQ(test, target) \
_ASSERT_MATRIX_EQ( \
    test, target, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_EQ" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_EQ(
    M<T> test, M<T> target,
    const char* file, int line, std::string message
) {
    _ASSERT_MATRIX_NEAR(test, target, static_cast<T>(0), file, line, message);
}

#define ASSERT_MATRIX_SAMESPARSITY(test, target, zero_tol) \
_ASSERT_MATRIX_SAMESPARSITY( \
    test, target, zero_tol, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_SAMESPARSITY" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_SAMESPARSITY(
    M<T> test, M<T> target, T zero_tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

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

#define ASSERT_MATRIX_ZERO(test, tol) \
_ASSERT_MATRIX_ZERO( \
    test, tol, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_ZERO" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_ZERO(
    M<T> test, T tol,
    const char* file, int line, std::string message
) {
    _ASSERT_MATRIX_NEAR(
        test, M<T>::Zero(test.rows(), test.cols()), tol, file, line, message
    );
}

#define ASSERT_MATRIX_IDENTITY(test, tol) \
_ASSERT_MATRIX_IDENTITY( \
    test, tol, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_IDENTITY" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_IDENTITY(
    M<T> test, T tol,
    const char* file, int line, std::string message
) {
    _ASSERT_MATRIX_NEAR(
        test, M<T>::Identity(test.rows(), test.cols()), tol, file, line, message
    );
}

#define ASSERT_MATRIX_LOWTRI(test, tol) \
_ASSERT_MATRIX_LOWTRI( \
    test, tol, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_LOWTRI" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_LOWTRI(
    M<T> test, T tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    for (int i=0; i<test.rows(); ++i) {
        for (int j=i+1; j<test.cols(); ++j) {
            ASSERT_NEAR(test.get_elem(i, j), static_cast<T>(0), tol);
        }
    }

}

#define ASSERT_MATRIX_UPPTRI(test, tol) \
_ASSERT_MATRIX_UPPTRI( \
    test, tol, __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_UPPTRI" \
)
template <template <typename> typename M, typename T>
void _ASSERT_MATRIX_UPPTRI(
    M<T> test, T tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    for (int i=0; i<test.rows(); ++i) {
        for (int j=0; j<i; ++j) {
            ASSERT_NEAR(test.get_elem(i, j), static_cast<T>(0), tol);
        }
    }

}

#endif