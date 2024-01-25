#ifndef SCALAR_H
#define SCALAR_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#include "tools/cuda_check.h"

template <typename T>
class Scalar
{
private:

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
    Scalar<Cast_T> cast();

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

    bool operator==(const Scalar& other) const;

};

#endif