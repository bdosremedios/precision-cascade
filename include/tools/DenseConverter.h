#ifndef DENSECONVERTER_H
#define DENSECONVERTER_H

#include "types/types.h"

template <template <typename> typename TMatrix, typename TPrecision>
class DenseConverter
{
public:

    DenseConverter() = default;

    TMatrix<TPrecision> convert_matrix(
        MatrixDense<TPrecision> mat
    ) const {
        std::runtime_error(
            "DenseConverter: reached unimplemented default convert_matrix "
            "implementation"
        );
    }

};

template <typename TPrecision>
class DenseConverter<MatrixDense, TPrecision>
{
public:

    MatrixDense<TPrecision> convert_matrix(
        MatrixDense<TPrecision> mat
    ) const {
        return mat;
    }

};

template <typename TPrecision>
class DenseConverter<Vector, TPrecision>
{
public:

    Vector<TPrecision> convert_matrix(
        MatrixDense<TPrecision> mat
    ) const {

        if (mat.cols() != 1) {
            throw std::runtime_error(
                "DenseConverter<Vector, TPrecision>: invalid csv for "
                "conversion in convert_matrix"
            );
        }

        return Vector<TPrecision>(mat.get_col(0));

    }

};

template <typename TPrecision>
class DenseConverter<NoFillMatrixSparse, TPrecision>
{
public:

    NoFillMatrixSparse<TPrecision> convert_matrix(
        MatrixDense<TPrecision> mat
    ) const {
        return NoFillMatrixSparse<TPrecision>(mat);
    }

};

#endif