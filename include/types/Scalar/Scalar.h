#ifndef SCALAR_H
#define SCALAR_H

#include "tools/cuda_check.h"
#include "tools/TypeIdentity.h"
#include "Scalar_gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <stdexcept>

template <typename TPrecision> class Vector;
template <typename TPrecision> class MatrixDense;
template <typename TPrecision> class NoFillMatrixSparse;

template <typename TPrecision>
class Scalar
{
private:

    template <typename> friend class Scalar;
    template <typename> friend class Vector;
    template <typename> friend class MatrixDense;
    template <typename> friend class NoFillMatrixSparse;

    TPrecision *d_scalar = nullptr;

    Scalar<__half> to_half() const;
    Scalar<float> to_float() const;
    Scalar<double> to_double() const;

    // Use argument overload for type specification rather than explicit
    // specialization due to limitation in g++
    Scalar<__half> cast(TypeIdentity<__half> _) const { return to_half(); }
    Scalar<float> cast(TypeIdentity<float> _) const { return to_float(); }
    Scalar<double> cast(TypeIdentity<double> _) const { return to_double(); }

public:

    Scalar() { check_cuda_error(cudaMalloc(&d_scalar, sizeof(TPrecision))); }
    Scalar(const TPrecision &val): Scalar() { set_scalar(val); }

    virtual ~Scalar() { check_cuda_error(cudaFree(d_scalar)); }

    Scalar<TPrecision> & operator=(const Scalar<TPrecision> &other) {
        if (this != &other) {
            check_cuda_error(cudaMemcpy(
                d_scalar,
                other.d_scalar,
                sizeof(TPrecision),
                cudaMemcpyDeviceToDevice
            ));
        }
        return *this;
    }

    Scalar(const Scalar &other): Scalar() { *this = other; }

    void set_scalar(const TPrecision &val) {
        check_cuda_error(cudaMemcpy(
            d_scalar,
            &val,
            sizeof(TPrecision),
            cudaMemcpyHostToDevice
        ));
    }
    TPrecision get_scalar() const {
        TPrecision h_scalar;
        check_cuda_error(cudaMemcpy(
            &h_scalar,
            d_scalar,
            sizeof(TPrecision),
            cudaMemcpyDeviceToHost
        ));
        return h_scalar;
    }

    template <typename Cast_TPrecision>
    Scalar<Cast_TPrecision> cast() const {
        return cast(TypeIdentity<Cast_TPrecision>());
    }

    Scalar<TPrecision> operator+(const Scalar& other) const;
    Scalar<TPrecision> operator-(const Scalar& other) const;

    Scalar<TPrecision> & operator+=(const Scalar& other);
    Scalar<TPrecision> & operator-=(const Scalar& other);

    Scalar<TPrecision> operator*(const Scalar& other) const;
    Scalar<TPrecision> operator/(const Scalar& other) const;

    Scalar<TPrecision> & operator*=(const Scalar& other);
    Scalar<TPrecision> & operator/=(const Scalar& other);

    Scalar<TPrecision> & abs();
    Scalar<TPrecision> & sqrt();
    Scalar<TPrecision> & reciprocol();

    bool operator==(const Scalar& other) const;

};

static inline Scalar<__half> SCALAR_ONE_H(static_cast<__half>(1.)); 
static inline Scalar<float> SCALAR_ONE_F(static_cast<float>(1.)); 
static inline Scalar<double> SCALAR_ONE_D(static_cast<double>(1.));

template <typename TPrecision>
class SCALAR_ONE
{ 
public:
    static Scalar<TPrecision> get();
};

static inline Scalar<__half> SCALAR_ZERO_H(static_cast<__half>(0.)); 
static inline Scalar<float> SCALAR_ZERO_F(static_cast<float>(0.)); 
static inline Scalar<double> SCALAR_ZERO_D(static_cast<double>(0.));

template <typename TPrecision>
class SCALAR_ZERO
{
public:
    static Scalar<TPrecision> get();
};

static inline Scalar<__half> SCALAR_MINUS_ONE_H(static_cast<__half>(-1.)); 
static inline Scalar<float> SCALAR_MINUS_ONE_F(static_cast<float>(-1.)); 
static inline Scalar<double> SCALAR_MINUS_ONE_D(static_cast<double>(-1.));

template <typename TPrecision>
class SCALAR_MINUS_ONE
{
public:
    static Scalar<TPrecision> get();
};

#endif