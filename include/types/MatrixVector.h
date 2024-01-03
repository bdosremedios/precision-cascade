#ifndef MATRIXVECTOR_H
#define MATRIXVECTOR_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T> class MatrixDense;
template <typename T> class MatrixSparse;

template <typename T>
class MatrixVector: private Matrix<T, Dynamic, 1>
{
private:

    using Parent = Matrix<T, Dynamic, 1>;

    static void check_n(int n) { if (n != 1) { throw std::runtime_error("Invalid number of columns for vector."); } }

    // Only allow MatrixDense and MatrixSparse to access private methods like base
    friend MatrixDense<T>;
    friend MatrixSparse<T>;
    const Parent &base() const { return *this; }

    int m;
    size_t mem_size;
    T *d_vec;

    void construction_helper() {
        cudaMalloc(&d_vec, mem_size);
    }

public:

    // *** Basic Constructors ***
    MatrixVector():
        m(0), mem_size(0)
    { construction_helper(); }

    MatrixVector(int arg_m, int arg_n):
        m(arg_m), mem_size(m*sizeof(T))
    { 
        check_n(n);
        construction_helper();
    }

    MatrixVector(int arg_m):
        m(arg_m), mem_size(m*sizeof(T))
    { construction_helper(); }

    MatrixVector(std::initializer_list<T> li):
        MatrixVector(li.size())
    {
        T *h_vec = static_cast<T *>(malloc(mem_size));

        int i=0;
        for (auto curr = std::cbegin(li); curr != std::cend(li); ++curr) {
            h_vec[i] = *curr;
            ++i;
        }

        cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
        free(h_vec);
    }

    // *** Conversion Constructors ***
    MatrixVector(const std::vector<T> &vec):
        m(vec.size()), mem_size(m*sizeof(T))
    {
        construction_helper();
        T *h_vec = static_cast<T *>(malloc(mem_size));
        for (int i=0; i<m; ++i) { h_vec[i] = vec[i]; }
        cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
        free(h_vec);
    }

    MatrixVector(const Parent &parent):
        Parent::Matrix(parent), m(parent.rows()), mem_size(m*sizeof(T))
    {
        check_n(parent.cols());
        construction_helper();

        T *h_vec = static_cast<T *>(malloc(mem_size));
        for (int i=0; i<m; ++i) { h_vec[i] = parent.coeff(i, 0); }
        cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
        free(h_vec);
    }

    MatrixVector(const typename MatrixDense<T>::Col &col):
        m(col.rows()), mem_size(m*sizeof(T))
    {
        construction_helper();

        T *h_vec = static_cast<T *>(malloc(mem_size));
        for (int i=0; i<m; ++i) { h_vec[i] = col.coeff(i, 0); }
        cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
        free(h_vec);
    }

    MatrixVector(const typename MatrixSparse<T>::Col &col):
        m(col.rows()), mem_size(m*sizeof(T))
    {
        construction_helper();

        T *h_vec = static_cast<T *>(malloc(mem_size));
        for (int i=0; i<m; ++i) { h_vec[i] = col.coeff(i, 0); }
        cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
        free(h_vec);
    }

    // *** Copy Constructor ***
    MatrixVector(const MatrixVector<T> &other) {
        *this = other;
    }

    // *** Destructor ***
    virtual ~MatrixVector() {
        cudaFree(d_vec);
        d_vec = nullptr;
    }

    // *** Copy-Assignment ***
    MatrixVector& operator=(const MatrixVector &other) {
        if (this != &other) {
            m = other.m;
            mem_size = other.mem_size;
            cudaFree(d_vec);
            cudaMalloc(&d_vec, mem_size);
            cudaMemcpy(d_vec, other.d_vec, mem_size, cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    // *** Static Creation ***
    static MatrixVector<T> Zero(int m, int n) {
        check_n(n);
        return typename Parent::Matrix(Parent::Zero(m, 1));
    }
    static MatrixVector<T> Zero(int m) { 

        std::cout << "Hit" << std::endl;
        // construction_helper();

        // T *h_vec = static_cast<T *>(malloc(mem_size));
        // for (int i=0; i<m; ++i) { h_vec[i] = col.coeff(i, 0); }
        // cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
        // free(h_vec);
        return typename Parent::Matrix(Parent::Zero(m, 1));
    }
    static MatrixVector<T> Ones(int m, int n) {
        check_n(n);
        return typename Parent::Matrix(Parent::Ones(m, 1));
    }
    static MatrixVector<T> Ones(int m) { return Ones(m, 1); }
    static MatrixVector<T> Random(int m, int n) {
        check_n(n);
        return typename Parent::Matrix(Parent::Random(m, 1));
    }
    static MatrixVector<T> Random(int m) { return Random(m, 1); }

    // *** Element Access ***

    const T get_elem(int row, int col) const {
        if (col > 0) { throw std::runtime_error("Invalid column access for vector."); }
        T h_elem;
        cudaMemcpy(&h_elem, d_vec+row, sizeof(T), cudaMemcpyDeviceToHost);
        return h_elem;
    }
    const T get_elem(int row) const { return get_elem(row, 0); }

    void set_elem(int row, int col, T val) {
        if (col > 0) { throw std::runtime_error("Invalid column access for vector."); }
        cudaMemcpy(d_vec+row, &val, sizeof(T), cudaMemcpyHostToDevice);
    }
    void set_elem(int row, T val) { set_elem(row, 0, val); }

    MatrixVector<T> slice(int start, int elements) const {
        T *h_vec = static_cast<T *>(malloc(elements*sizeof(T)));
        cublasGetVector(elements, sizeof(T), d_vec+start, 1, h_vec, 1);
        std::vector<T> vec_temp;
        for (int i=0; i<elements; ++i) { vec_temp.push_back(h_vec[i]); }
        free(h_vec);
        return MatrixVector(vec_temp);
    }

    // *** Properties ***
    int rows() const { return m; }
    int cols() const { return 1; }
    void print() const {
        T *h_vec = static_cast<T *>(malloc(mem_size));
        cublasGetVector(m, sizeof(T), d_vec, 1, h_vec, 1);
        for (int i=0; i<m; ++i) {
            std::cout << static_cast<double>(h_vec[i]) << std::endl;
        }
        std::cout << std::endl;
        std::cout << Parent::rows() << " " << Parent::cols() << std::endl << std::endl;
        free(h_vec);
    }

    // *** Resizing ***
    void reduce() { ; }

    // *** Boolean ***
    bool operator==(const MatrixVector<T> &rhs) const { return Parent::isApprox(rhs); }

    // *** Explicit Cast ***
    template <typename Cast_T>
    MatrixVector<Cast_T> cast() const {
        return typename Matrix<Cast_T, Dynamic, 1>::Matrix(Parent::template cast<Cast_T>());
    }

    // *** Arithmetic and Compound Operations ***
    T dot(const MatrixVector<T> &vec) const { return Parent::dot(vec); }
    T norm() const { return Parent::norm(); }
    MatrixVector<T> operator*(const T &scalar) const {
        return typename Parent::Matrix(Parent::operator*(scalar));
    }
    MatrixVector<T> operator/(const T &scalar) const {
        return typename Parent::Matrix(Parent::operator/(scalar));
    }
    MatrixVector<T> operator-(const MatrixVector<T> &vec) const {
        return typename Parent::Matrix(Parent::operator-(vec));
    }
    MatrixVector<T> operator+(const MatrixVector<T> &vec) const {
        return typename Parent::Matrix(Parent::operator+(vec));
    }
    MatrixVector<T> operator+=(const MatrixVector<T> &vec) { return *this = *this + vec; };
    MatrixVector<T> operator-=(const MatrixVector<T> &vec) { return *this = *this - vec; };
};

#include "MatrixDense.h"
#include "MatrixSparse.h"

#endif