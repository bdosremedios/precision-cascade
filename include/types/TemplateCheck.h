#ifndef TEMPLATECHECK_H
#define TEMPLATECHECK_H

#include "MatrixDense.h"
#include "MatrixSparse.h"
#include "MatrixVector.h"

// https://stackoverflow.com/questions/38630445/stdis-same-equivalent-for-unspecialised-template-types
template <template <typename> typename, template<typename...> typename> 
struct is_same_template : std::false_type{};

template <template <typename> typename T>
struct is_same_template<T,T> : std::true_type{};

template <template <typename> typename M>
void assert_valid_type() {
    static_assert(
        ((is_same_template<M, MatrixSparse>::value) || (is_same_template<M, MatrixDense>::value)),
        "M argument must be type MatrixSparse or MatrixDense"
    );
}

template <template <typename> typename M>
void assert_valid_type_or_vec() {
    static_assert(
        ((is_same_template<M, MatrixSparse>::value) ||
         (is_same_template<M, MatrixDense>::value) ||
         (is_same_template<M, MatrixVector>::value)),
        "M argument must be type MatrixSparse or MatrixDense or MatrixVector"
    );
}

#endif