#ifndef SCALAR_CUH
#define SCALAR_CUH

#include <cuda_runtime.h>

#include "tools/cuda_check.h"
#include "types/Scalar.h"

template <typename T>
__global__ void scalar_add(T *scalar_1, T *scalar_2, T *result) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]+scalar_2[tid];
}

template <typename T>
Scalar<T> Scalar<T>::operator+(const Scalar<T> &other) const {
    Scalar<T> result;
    scalar_add<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename T>
Scalar<T> & Scalar<T>::operator+=(const Scalar<T> &other) {
    scalar_add<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename T>
__global__ void scalar_minus(T *scalar_1, T *scalar_2, T *result) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]-scalar_2[tid];
}

template <typename T>
Scalar<T> Scalar<T>::operator-(const Scalar<T> &other) const {
    Scalar<T> result;
    scalar_minus<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename T>
Scalar<T> & Scalar<T>::operator-=(const Scalar<T> &other) {
    scalar_minus<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename T>
__global__ void scalar_mult(T *scalar_1, T *scalar_2, T *result) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]*scalar_2[tid];
}

template <typename T>
Scalar<T> Scalar<T>::operator*(const Scalar<T> &other) const {
    Scalar<T> result;
    scalar_mult<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename T>
Scalar<T> & Scalar<T>::operator*=(const Scalar<T> &other) {
    scalar_mult<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename T>
__global__ void scalar_div(T *scalar_1, T *scalar_2, T *result) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid]/scalar_2[tid];
}

template <typename T>
Scalar<T> Scalar<T>::operator/(const Scalar<T> &other) const {
    Scalar<T> result;
    scalar_div<<<1, 1>>>(d_scalar, other.d_scalar, result.d_scalar);
    return result;
}

template <typename T>
Scalar<T> & Scalar<T>::operator/=(const Scalar<T> &other) {
    scalar_div<<<1, 1>>>(d_scalar, other.d_scalar, d_scalar);
    return *this;
}

template <typename T>
__global__ void scalar_bool_eq(T *scalar_1, T *scalar_2, bool *result) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[tid] = scalar_1[tid] == scalar_2[tid];
}

template <typename T>
bool Scalar<T>::operator==(const Scalar<T> &other) const {
    if (this == &other) {
        return true;
    } else {
        bool result;
        bool *d_result;
        check_cuda_error(cudaMalloc(&d_result, sizeof(bool)));

        scalar_bool_eq<<<1, 1>>>(d_scalar, other.d_scalar, d_result);

        check_cuda_error(cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
        check_cuda_error(cudaFree(d_result));

        return result;
    }
}

#endif