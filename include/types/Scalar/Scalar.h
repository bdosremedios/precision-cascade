#ifndef SCALAR_H
#define SCALAR_H

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "tools/cuda_check.h"

#include "Scalar_gpu_kernels.cuh"

template <typename T> class Vector;
template <typename T> class MatrixDense;
template <typename T> class NoFillMatrixSparse;

template <typename T>
class Scalar
{
private:

    template <typename> friend class Scalar;
    template <typename> friend class Vector;
    template <typename> friend class MatrixDense;
    template <typename> friend class NoFillMatrixSparse;

    T *d_scalar = nullptr;

    Scalar<__half> to_half() const;
    Scalar<float> to_float() const;
    Scalar<double> to_double() const;

public:

    // *** Constructors ***
    Scalar() { check_cuda_error(cudaMalloc(&d_scalar, sizeof(T))); }
    Scalar(const T &val): Scalar() { set_scalar(val); }

    // *** Destructor ***
    virtual ~Scalar() { check_cuda_error(cudaFree(d_scalar)); }

    // *** Copy Assignment ***
    Scalar<T> & operator=(const Scalar<T> &other) {
        if (this != &other) {
            check_cuda_error(cudaMemcpy(
                d_scalar, other.d_scalar, sizeof(T), cudaMemcpyDeviceToDevice
            ));
        }
        return *this;
    }

    // *** Copy Constructor ***
    Scalar(const Scalar &other): Scalar() { *this = other; }

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
    Scalar<Cast_T> cast() const {
        throw std::runtime_error("Scalar: invalid cast conversion");
    }

    template <> Scalar<__half> cast<__half>() const { return to_half(); }
    template <> Scalar<float> cast<float>() const { return to_float(); }
    template <> Scalar<double> cast<double>() const { return to_double(); }

    // *** Operations ***
    Scalar<T> operator+(const Scalar& other) const;
    Scalar<T> operator-(const Scalar& other) const;

    Scalar<T> & operator+=(const Scalar& other);
    Scalar<T> & operator-=(const Scalar& other);

    Scalar<T> operator*(const Scalar& other) const;
    Scalar<T> operator/(const Scalar& other) const;

    Scalar<T> & operator*=(const Scalar& other);
    Scalar<T> & operator/=(const Scalar& other);

    Scalar<T> & abs();
    Scalar<T> & sqrt();
    Scalar<T> & reciprocol();

    bool operator==(const Scalar& other) const;

};

static inline Scalar<__half> SCALAR_ONE_H(static_cast<__half>(1.)); 
static inline Scalar<float> SCALAR_ONE_F(static_cast<float>(1.)); 
static inline Scalar<double> SCALAR_ONE_D(static_cast<double>(1.));

template <typename T>
class SCALAR_ONE
{ 
public:
    static Scalar<T> get();
};

static inline Scalar<__half> SCALAR_ZERO_H(static_cast<__half>(0.)); 
static inline Scalar<float> SCALAR_ZERO_F(static_cast<float>(0.)); 
static inline Scalar<double> SCALAR_ZERO_D(static_cast<double>(0.));

template <typename T>
class SCALAR_ZERO
{
public:
    static Scalar<T> get();
};

static inline Scalar<__half> SCALAR_MINUS_ONE_H(static_cast<__half>(-1.)); 
static inline Scalar<float> SCALAR_MINUS_ONE_F(static_cast<float>(-1.)); 
static inline Scalar<double> SCALAR_MINUS_ONE_D(static_cast<double>(-1.));

template <typename T>
class SCALAR_MINUS_ONE
{
public:
    static Scalar<T> get();
};

#endif