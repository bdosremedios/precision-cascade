#ifndef DENSECONVERTER_H
#define DENSECONVERTER_H

#include "types/types.h"

template <template <typename> typename M, typename T>
class DenseConverter
{
public:

    DenseConverter() = default;

    M<T> convert_matrix(MatrixDense<T> mat) const {
        std::runtime_error("DenseConverter: reached unimplemented default convert_matrix implementation");
    }

};

template <typename T>
class DenseConverter<MatrixDense, T>
{
public:

    MatrixDense<T> convert_matrix(MatrixDense<T> mat) const {
        return mat;
    }

};

template <typename T>
class DenseConverter<Vector, T>
{
public:

    Vector<T> convert_matrix(MatrixDense<T> mat) const {

        if (mat.cols() != 1) {
            throw std::runtime_error(
                "DenseConverter<Vector, T>: invalid csv for conversion to Vector in convert_matrix"
            );
        }

        return Vector<T>(mat.get_col(0));

    }

};

template <typename T>
class DenseConverter<NoFillMatrixSparse, T>
{
public:

    NoFillMatrixSparse<T> convert_matrix(MatrixDense<T> mat) const {
        return NoFillMatrixSparse<T>(mat);
    }

};

#endif