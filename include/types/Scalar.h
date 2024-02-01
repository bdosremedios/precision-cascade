#ifndef SCALAR_H
#define SCALAR_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>

#include "tools/cuda_check.h"

template <typename T> class Vector;
template <typename T> class MatrixDense;

template <typename T>
class Scalar
{
private:

    template <typename> friend class Scalar;
    template <typename> friend class Vector;
    template <typename> friend class MatrixDense;

    T *d_scalar = nullptr;

public:

    // *** Constructors ***
    Scalar() { check_cuda_error(cudaMalloc(&d_scalar, sizeof(T))); }
    Scalar(const T &val): Scalar() { set_scalar(val); }

    // *** Destructor/Copy-Constructor/Assignment-Constructor ***
    virtual ~Scalar() { check_cuda_error(cudaFree(d_scalar)); }
    Scalar(const Scalar &other): Scalar() { *this = other; }
    Scalar<T> & operator=(const Scalar<T> &other) {
        if (this != &other) {
            check_cuda_error(cudaMemcpy(d_scalar, other.d_scalar, sizeof(T), cudaMemcpyDeviceToDevice));
        }
        return *this;
    }

    // *** Access ***
    void set_scalar(const T &val) {
        check_cuda_error(cudaMemcpy(d_scalar, &val, sizeof(T), cudaMemcpyHostToDevice));
    }
    T get_scalar() const {
        T h_scalar;
        check_cuda_error(cudaMemcpy(&h_scalar, d_scalar, sizeof(T), cudaMemcpyDeviceToHost));
        return h_scalar;
    }
    void print() { std::cout << static_cast<double>(get_scalar()) << std::endl; }

    // *** Cast ***
    template <typename Cast_T>
    Scalar<Cast_T> cast() const { throw std::runtime_error("Scalar: invalid cast conversion"); }

    Scalar<__half> to_half() const;
    template <> Scalar<__half> cast<__half>() const { return to_half(); }
    Scalar<float> to_float() const;
    template <> Scalar<float> cast<float>() const { return to_float(); }
    Scalar<double> to_double() const;
    template <> Scalar<double> cast<double>() const { return to_double(); }

    // *** Arithmetic/Compound Operations ***
    Scalar<T> operator+(const Scalar& other) const;
    Scalar<T> operator-(const Scalar& other) const;

    void operator+=(const Scalar& other);
    void operator-=(const Scalar& other);

    Scalar<T> operator*(const Scalar& other) const;
    Scalar<T> operator/(const Scalar& other) const;

    void operator*=(const Scalar& other);
    void operator/=(const Scalar& other);

    Scalar<T> & abs();
    Scalar<T> & sqrt();
    Scalar<T> & reciprocol();

    bool operator==(const Scalar& other) const;

};

#include "Vector.h"
#include "MatrixDense.h"

#endif