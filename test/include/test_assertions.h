#ifndef TEST_ASSERTIONS_H
#define TEST_ASSERTIONS_H

#include "tools/cuda_check.h"
#include "tools/abs.h"
#include "types/types.h"

#include "gtest/gtest.h"

#include <functional>
#include <string>

using namespace cascade;

static bool *print_errors;

#define CHECK_FUNC_HAS_RUNTIME_ERROR(to_print, func) \
_CHECK_FUNC_HAS_RUNTIME_ERROR( \
    to_print, func, \
    __FILE__, __LINE__, "AssertionError for CHECK_FUNC_HAS_RUNTIME_ERROR" \
)

inline void _CHECK_FUNC_HAS_RUNTIME_ERROR(
    bool *to_print, std::function<void()> func,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    try {
        func();
        FAIL();
    } catch (std::runtime_error &e) {
        if (*to_print) {
            std::cout << e.what() << std::endl;
        }
    }

}

#define ASSERT_VECTOR_NEAR(test, target, tol) \
_ASSERT_VECTOR_NEAR( \
    test, target, tol, \
    __FILE__, __LINE__, "AssertionError for ASSERT_VECTOR_NEAR" \
)

template <typename TPrecision>
void _ASSERT_VECTOR_NEAR(
    const Vector<TPrecision> &test,
    const Vector<TPrecision> &target,
    TPrecision tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());

    for (int i=0; i<target.rows(); ++i) {
        ASSERT_NEAR(
            test.get_elem(i).get_scalar(),
            target.get_elem(i).get_scalar(),
            tol
        );
    }

}

template <typename TPrecision>
void _ASSERT_VECTOR_NEAR(
    const Vector<TPrecision> &test,
    const Vector<TPrecision> &target,
    const Vector<TPrecision> &abs_tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(abs_tol.rows(), target.rows());

    for (int i=0; i<target.rows(); ++i) {
        ASSERT_NEAR(
            test.get_elem(i).get_scalar(),
            target.get_elem(i).get_scalar(),
            abs_tol.get_elem(i).get_scalar()
        );
    }

}

#define ASSERT_VECTOR_ZERO(test, tol) \
_ASSERT_VECTOR_ZERO( \
    test, tol, \
    __FILE__, __LINE__, "AssertionError for _ASSERT_VECTOR_ZERO" \
)
template <typename TPrecision>
void _ASSERT_VECTOR_ZERO(
    const Vector<TPrecision> &test,
    TPrecision tol,
    const char* file, int line, std::string message
) {
    _ASSERT_VECTOR_NEAR(
        test,
        Vector<TPrecision>::Zero(
            test.get_cu_handles(),
            test.rows(),
            test.cols()
        ),
        tol,
        file, line, message
    );
}

#define ASSERT_VECTOR_EQ(test, target) \
_ASSERT_VECTOR_EQ( \
    test, target, \
    __FILE__, __LINE__, "AssertionError for ASSERT_VECTOR_EQ" \
)

template <typename TPrecision>
void _ASSERT_VECTOR_EQ(
    const Vector<TPrecision> &test,
    const Vector<TPrecision> &target,
    const char* file, int line, std::string message
) {
    _ASSERT_VECTOR_NEAR(
        test, target, static_cast<TPrecision>(0), file, line, message
    );
}

#define ASSERT_MATRIX_NEAR(test, target, tol) \
_ASSERT_MATRIX_NEAR( \
    test, target, tol, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_NEAR" \
)

template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_NEAR(
    const TMatrix<TPrecision> &test,
    const TMatrix<TPrecision> &target,
    TPrecision tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_NEAR(
                test.get_elem(i, j).get_scalar(),
                target.get_elem(i, j).get_scalar(),
                tol
            );
        }
    }

}

template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_NEAR(
    const TMatrix<TPrecision> &test,
    const TMatrix<TPrecision> &target,
    const TMatrix<TPrecision> abs_tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());
    ASSERT_EQ(abs_tol.rows(), target.rows());
    ASSERT_EQ(abs_tol.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_NEAR(
                test.get_elem(i, j).get_scalar(),
                target.get_elem(i, j).get_scalar(),
                abs_tol.get_elem(i, j).get_scalar()
            );
        }
    }

}

#define ASSERT_MATRIX_LT(test, target) \
_ASSERT_MATRIX_LT( \
    test, target, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_LT" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_LT(
    const TMatrix<TPrecision> &test,
    const TMatrix<TPrecision> &target,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_LT(
                test.get_elem(i, j).get_scalar(),
                target.get_elem(i, j).get_scalar()
            );
        }
    }

}

#define ASSERT_MATRIX_GT(test, target) \
_ASSERT_MATRIX_GT( \
    test, target, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_GT" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_GT(
    const TMatrix<TPrecision> &test,
    const TMatrix<TPrecision> &target,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_GT(
                test.get_elem(i, j).get_scalar(),
                target.get_elem(i, j).get_scalar()
            );
        }
    }

}

#define ASSERT_MATRIX_EQ(test, target) \
_ASSERT_MATRIX_EQ( \
    test, target, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_EQ" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_EQ(
    const TMatrix<TPrecision> &test,
    const TMatrix<TPrecision> &target,
    const char* file, int line, std::string message
) {
    _ASSERT_MATRIX_NEAR(
        test, target, static_cast<TPrecision>(0), file, line, message
    );
}

#define ASSERT_MATRIX_SAMESPARSITY(test, target, zero_tol) \
_ASSERT_MATRIX_SAMESPARSITY( \
    test, target, zero_tol, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_SAMESPARSITY" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_SAMESPARSITY(
    const TMatrix<TPrecision> &test,
    const TMatrix<TPrecision> &target,
    TPrecision zero_tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            if (abs_ns::abs(target.get_elem(i, j).get_scalar()) <= zero_tol) {
                ASSERT_LE(
                    abs_ns::abs(test.get_elem(i, j).get_scalar()),
                    zero_tol
                );
            }
        }
    }

}

#define ASSERT_MATRIX_ZERO(test, tol) \
_ASSERT_MATRIX_ZERO( \
    test, tol, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_ZERO" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_ZERO(
    const TMatrix<TPrecision> &test,
    TPrecision tol,
    const char* file, int line, std::string message
) {
    _ASSERT_MATRIX_NEAR(
        test,
        TMatrix<TPrecision>::Zero(
            test.get_cu_handles(),
            test.rows(),
            test.cols()
        ),
        tol,
        file, line, message
    );
}

#define ASSERT_MATRIX_IDENTITY(test, tol) \
_ASSERT_MATRIX_IDENTITY( \
    test, tol, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_IDENTITY" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_IDENTITY(
    const TMatrix<TPrecision> &test,
    TPrecision tol,
    const char* file, int line, std::string message
) {
    _ASSERT_MATRIX_NEAR(
        test,
        TMatrix<TPrecision>::Identity(
            test.get_cu_handles(),
            test.rows(),
            test.cols()
        ),
        tol, file, line, message
    );
}

#define ASSERT_MATRIX_LOWTRI(test, tol) \
_ASSERT_MATRIX_LOWTRI( \
    test, tol, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_LOWTRI" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_LOWTRI(
    const TMatrix<TPrecision> &test,
    TPrecision tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    for (int i=0; i<test.rows(); ++i) {
        for (int j=i+1; j<test.cols(); ++j) {
            ASSERT_EQ(
                test.get_elem(i, j).get_scalar(),
                static_cast<TPrecision>(0)
            );
        }
    }

}

#define ASSERT_MATRIX_UPPTRI(test, tol) \
_ASSERT_MATRIX_UPPTRI( \
    test, tol, \
    __FILE__, __LINE__, "AssertionError for ASSERT_MATRIX_UPPTRI" \
)
template <template <typename> typename TMatrix, typename TPrecision>
void _ASSERT_MATRIX_UPPTRI(
    const TMatrix<TPrecision> &test,
    TPrecision tol,
    const char* file, int line, std::string message
) {

    testing::ScopedTrace scope(file, line, message);

    for (int i=0; i<test.rows(); ++i) {
        for (int j=0; j<i; ++j) {
            ASSERT_EQ(
                test.get_elem(i, j).get_scalar(),
                static_cast<TPrecision>(0)
            );
        }
    }

}

#endif