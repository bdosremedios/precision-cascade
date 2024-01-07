#ifndef MATRIXVECTOR_H
#define MATRIXVECTOR_H

#include <Eigen/Dense>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// using Eigen::Matrix;
// using Eigen::Dynamic;

template <typename T> class MatrixDense;
template <typename T> class MatrixSparse;

template <typename T>
class MatrixVector //: private Matrix<T, Dynamic, 1>
{
private:

    // using Parent = Matrix<T, Dynamic, 1>;

    static void check_n(int n) {
        if (n != 1) { throw std::runtime_error("Invalid number of columns for vector."); }
    }

    // Allow all similar type Matrices and different type variants of self to access private methods
    template <typename> friend class MatrixVector;
    // friend MatrixDense<T>;
    // friend MatrixSparse<T>;
    // const Parent &base() const { return *this; }

    int m;
    size_t mem_size;
    T *d_vec = nullptr;
    cublasHandle_t handle;

    void allocate_d_vec() {
        cudaMalloc(&d_vec, mem_size);
    }

    MatrixVector(const cublasHandle_t &arg_handle, const T *h_vec, const int m_elem):
        handle(arg_handle), m(m_elem), mem_size(m*sizeof(T))
    {
        allocate_d_vec();
        cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
    }

public:

    // *** Basic Constructors ***
    MatrixVector(const cublasHandle_t &arg_handle):
        handle(arg_handle), m(0), mem_size(0)
    { allocate_d_vec(); }

    MatrixVector(const cublasHandle_t &arg_handle, int arg_m, int arg_n):
        handle(arg_handle), m(arg_m), mem_size(m*sizeof(T))
    { 
        check_n(n);
        allocate_d_vec();
    }

    MatrixVector(const cublasHandle_t &arg_handle, int arg_m):
        handle(arg_handle), m(arg_m), mem_size(m*sizeof(T))
    {
        allocate_d_vec();
    }

    MatrixVector(const cublasHandle_t &arg_handle, std::initializer_list<T> li):
        MatrixVector(arg_handle, li.size())
    {

        T *h_vec = static_cast<T *>(malloc(mem_size));

        struct loop_vars { int i; typename std::initializer_list<T>::const_iterator curr; };
        for (loop_vars vars = {0, std::cbegin(li)}; vars.curr != std::cend(li); ++vars.curr, ++vars.i) {
            h_vec[vars.i] = *vars.curr;
        }

        cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
        free(h_vec);

    }

    // *** Conversion Constructors ***
    // MatrixVector(const cublasHandle_t &arg_handle, const Parent &parent):
    //     handle(arg_handle), Parent::Matrix(parent), m(parent.rows()), mem_size(m*sizeof(T))
    // {

    //     check_n(parent.cols());
    //     allocate_d_vec();

    //     T *h_vec = static_cast<T *>(malloc(mem_size));
    //     for (int i=0; i<m; ++i) { h_vec[i] = parent.coeff(i, 0); }
    //     cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
    //     free(h_vec);

    // }

    // MatrixVector(const cublasHandle_t &arg_handle, const typename MatrixDense<T>::Col &col):
    //     handle(arg_handle), m(col.rows()), mem_size(m*sizeof(T))
    // {

    //     allocate_d_vec();

    //     T *h_vec = static_cast<T *>(malloc(mem_size));
    //     for (int i=0; i<m; ++i) { h_vec[i] = col.coeff(i, 0); }
    //     cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
    //     free(h_vec);

    // }

    // MatrixVector(const cublasHandle_t &arg_handle, const typename MatrixSparse<T>::Col &col):
    //     handle(arg_handle), m(col.rows()), mem_size(m*sizeof(T))
    // {

    //     allocate_d_vec();

    //     T *h_vec = static_cast<T *>(malloc(mem_size));
    //     for (int i=0; i<m; ++i) { h_vec[i] = col.coeff(i, 0); }
    //     cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
    //     free(h_vec);

    // }

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
            handle = other.handle;
            m = other.m;
            mem_size = other.mem_size;
            cudaFree(d_vec);
            cudaMalloc(&d_vec, mem_size);
            cudaMemcpy(d_vec, other.d_vec, mem_size, cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    // *** Static Creation ***
    static MatrixVector<T> Zero(const cublasHandle_t &arg_handle, int m) {

        T *h_vec = static_cast<T *>(malloc(m*sizeof(T)));

        for (int i=0; i<m; ++i) { h_vec[i] = static_cast<T>(0); }
        MatrixVector<T> created_vec(arg_handle, h_vec, m);

        free(h_vec);

        return created_vec;

    }

    static MatrixVector<T> Zero(const cublasHandle_t &arg_handle, int m, int n) {
        check_n(n);
        return Zero(arg_handle, m);
    }

    static MatrixVector<T> Ones(const cublasHandle_t &arg_handle, int m) {

        T *h_vec = static_cast<T *>(malloc(m*sizeof(T)));

        for (int i=0; i<m; ++i) { h_vec[i] = static_cast<T>(1); }
        MatrixVector<T> created_vec(arg_handle, h_vec, m);

        free(h_vec);

        return created_vec;

    }

    static MatrixVector<T> Ones(const cublasHandle_t &arg_handle, int m, int n) {
        check_n(n);
        return Ones(arg_handle, m);
    }

    // Needed for testing (don't need to optimize performance)
    static MatrixVector<T> Random(const cublasHandle_t &arg_handle, int m) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        T *h_vec = static_cast<T *>(malloc(m*sizeof(T)));

        for (int i=0; i<m; ++i) { h_vec[i] = static_cast<T>(dist(gen)); }
        MatrixVector<T> created_vec(arg_handle, h_vec, m);

        free(h_vec);

        return created_vec;

    }

    // Needed for testing (don't need to optimize performance)
    static MatrixVector<T> Random(const cublasHandle_t &arg_handle, int m, int n) {
        check_n(n);
        return Random(arg_handle, m);
    }

    // *** Element Access ***
    const T get_elem(int row, int col) const {
        if (col != 0) { throw std::runtime_error("Invalid vector column access"); }
        if ((row < 0) || (row >= m)) { throw std::runtime_error("Invalid vector row access"); }
        T h_elem;
        cudaMemcpy(&h_elem, d_vec+row, sizeof(T), cudaMemcpyDeviceToHost);
        return h_elem;
    }
    const T get_elem(int row) const { return get_elem(row, 0); }

    void set_elem(int row, int col, T val) {
        if (col != 0) { throw std::runtime_error("Invalid vector column access"); }
        if ((row < 0) || (row >= m)) { throw std::runtime_error("Invalid vector row access"); }
        cudaMemcpy(d_vec+row, &val, sizeof(T), cudaMemcpyHostToDevice);
    }
    void set_elem(int row, T val) { set_elem(row, 0, val); }

    MatrixVector<T> slice(int start, int m_elem) const {

        T *h_vec = static_cast<T *>(malloc(m_elem*sizeof(T)));

        cublasGetVector(m_elem, sizeof(T), d_vec+start, 1, h_vec, 1);
        MatrixVector<T> created_vec(handle, h_vec, m_elem);

        free(h_vec);

        return created_vec;

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

        free(h_vec);

    }

    // *** Resizing ***
    void reduce() { ; } // Leave empty MatrixVector is dense

    // *** Boolean ***
    bool operator==(const MatrixVector<T> &other) const {
        
        if (this == &other) { return true; }
        if (m != other.m) { return false; }

        T *h_vec_self = static_cast<T *>(malloc(mem_size));
        T *h_vec_other = static_cast<T *>(malloc(mem_size));

        cublasGetVector(m, sizeof(T), d_vec, 1, h_vec_self, 1);
        cublasGetVector(m, sizeof(T), other.d_vec, 1, h_vec_other, 1);

        bool is_equal = true;
        for (int i=0; i<m; ++i) { is_equal = is_equal && (h_vec_self[i] == h_vec_other[i]); }

        free(h_vec_self);
        free(h_vec_other);

        return is_equal;

    }

    // *** Explicit Cast ***
    template <typename Cast_T>
    MatrixVector<Cast_T> cast() const {
        
        T *h_vec = static_cast<T *>(malloc(mem_size));
        Cast_T *h_cast_vec = static_cast<Cast_T *>(malloc(m*sizeof(Cast_T)));

        cublasGetVector(m, sizeof(T), d_vec, 1, h_vec, 1);
        for (int i=0; i<m; ++i) { h_cast_vec[i] = static_cast<Cast_T>(h_vec[i]); }
        MatrixVector<Cast_T> created_vec(handle, h_cast_vec, m);

        free(h_vec);
        free(h_cast_vec);

        return created_vec;
    
    }

    // *** Arithmetic and Compound Operations ***
    MatrixVector<T> operator*(const T &scalar) const;
    MatrixVector<T> operator/(const T &scalar) const {
        return operator*(static_cast<T>(1.)/scalar);
    }

    MatrixVector<T> & operator*=(const T &scalar);
    MatrixVector<T> & operator/=(const T &scalar) {
        return operator*=(static_cast<T>(1.)/scalar);
    }

    MatrixVector<T> operator-(const MatrixVector<T> &vec) const;
    MatrixVector<T> operator+(const MatrixVector<T> &vec) const;

    MatrixVector<T> & operator+=(const MatrixVector<T> &vec);
    MatrixVector<T> & operator-=(const MatrixVector<T> &vec);
    
    T dot(const MatrixVector<T> &vec) const;

    T norm() const;

};

#include "MatrixDense.h"
#include "MatrixSparse.h"

#endif