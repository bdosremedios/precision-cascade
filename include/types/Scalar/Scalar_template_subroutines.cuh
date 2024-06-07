#ifndef SCALAR_CUH
#define SCALAR_CUH

#include "tools/cuda_check.h"
#include "Scalar.h"

#include <cuda_runtime.h>

template <typename TPrecision>
__global__ void scalar_add(
    TPrecision *scalar_1, TPrecision *scalar_2, TPrecision *result
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]+scalar_2[tid];
}

template <typename TPrecision>
Scalar<TPrecision> Scalar<TPrecision>::operator+(
    const Scalar<TPrecision> &other
) const {
    Scalar<TPrecision> result;
    scalar_add<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename TPrecision>
Scalar<TPrecision> & Scalar<TPrecision>::operator+=(
    const Scalar<TPrecision> &other
) {
    scalar_add<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename TPrecision>
__global__ void scalar_minus(
    TPrecision *scalar_1, TPrecision *scalar_2, TPrecision *result
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]-scalar_2[tid];
}

template <typename TPrecision>
Scalar<TPrecision> Scalar<TPrecision>::operator-(
    const Scalar<TPrecision> &other
) const {
    Scalar<TPrecision> result;
    scalar_minus<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename TPrecision>
Scalar<TPrecision> & Scalar<TPrecision>::operator-=(
    const Scalar<TPrecision> &other
) {
    scalar_minus<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename TPrecision>
__global__ void scalar_mult(
    TPrecision *scalar_1, TPrecision *scalar_2, TPrecision *result
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]*scalar_2[tid];
}

template <typename TPrecision>
Scalar<TPrecision> Scalar<TPrecision>::operator*(
    const Scalar<TPrecision> &other
) const {
    Scalar<TPrecision> result;
    scalar_mult<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename TPrecision>
Scalar<TPrecision> & Scalar<TPrecision>::operator*=(
    const Scalar<TPrecision> &other
) {
    scalar_mult<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename TPrecision>
__global__ void scalar_div(
    TPrecision *scalar_1, TPrecision *scalar_2, TPrecision *result
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]/scalar_2[tid];
}

template <typename TPrecision>
Scalar<TPrecision> Scalar<TPrecision>::operator/(
    const Scalar<TPrecision> &other
) const {
    Scalar<TPrecision> result;
    scalar_div<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename TPrecision>
Scalar<TPrecision> & Scalar<TPrecision>::operator/=(
    const Scalar<TPrecision> &other
) {
    scalar_div<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename TPrecision>
__global__ void scalar_bool_eq(
    TPrecision *scalar_1, TPrecision *scalar_2, bool *result
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid] == scalar_2[tid];
}

template <typename TPrecision>
bool Scalar<TPrecision>::operator==(const Scalar<TPrecision> &other) const {
    if (this == &other) {
        return true;
    } else {
        bool result;
        bool *d_result;
        check_cuda_error(cudaMalloc(&d_result, sizeof(bool)));

        scalar_bool_eq<<<1, 1>>>(d_scalar, other.d_scalar, d_result);

        check_cuda_error(cudaMemcpy(
            &result, d_result, sizeof(bool), cudaMemcpyDeviceToHost
        ));
        check_cuda_error(cudaFree(d_result));

        return result;
    }
}

#endif