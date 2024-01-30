#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <random>
#include <initializer_list>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tools/cuda_check.h"
#include "tools/vector_sort.h"

template <typename T> class MatrixDense;
template <typename T> class MatrixSparse;

template <typename T>
class Vector
{
private:

    static void check_n(int n) {
        if (n != 1) { throw std::runtime_error("Vector: invalid number of columns for vector"); }
    }

    // Allow all similar type Matrices and different type variants of self to access private methods
    template <typename> friend class Vector;
    friend MatrixDense<T>;
    friend MatrixDense<T>::Block;
    // friend MatrixSparse<T>;

    cublasHandle_t handle;
    int m_rows;
    size_t mem_size;
    T *d_vec = nullptr;

    void allocate_d_vec() {
        check_cuda_error(cudaMalloc(&d_vec, mem_size));
    }

    void check_vecvec_op_compatibility(const Vector<T> &other) const {
        if (m_rows != other.m_rows) {
            throw std::runtime_error("Vector: incompatible vector sizes for vec-vec operation");
        }
    }

public:

    // *** Constructors ***
    Vector(const cublasHandle_t &arg_handle, int arg_m, int arg_n):
        handle(arg_handle), m_rows(arg_m), mem_size(m_rows*sizeof(T))
    { 
        check_n(arg_n);
        allocate_d_vec();
    }

    Vector(const cublasHandle_t &arg_handle, int arg_m): Vector(arg_handle, arg_m, 1) {}
    Vector(const cublasHandle_t &arg_handle): Vector(arg_handle, 0) {}

    Vector(const cublasHandle_t &arg_handle, std::initializer_list<T> li):
        Vector(arg_handle, li.size())
    {
        T *h_vec = static_cast<T *>(malloc(mem_size));

        struct loop_vars { int i; typename std::initializer_list<T>::const_iterator curr; };
        for (loop_vars vars = {0, std::cbegin(li)}; vars.curr != std::cend(li); ++vars.curr, ++vars.i) {
            h_vec[vars.i] = *vars.curr;
        }
        if (m_rows > 0) {
            check_cublas_status(cublasSetVector(m_rows, sizeof(T), h_vec, 1, d_vec, 1));
        }

        free(h_vec);
    }

    // *** Dynamic Memory *** (assumes outer code handles dynamic memory properly)
    Vector(const cublasHandle_t &arg_handle, const T *h_vec, const int m_elem):
        Vector(arg_handle, m_elem)
    {
        if (m_elem > 0) {
            check_cublas_status(cublasSetVector(m_rows, sizeof(T), h_vec, 1, d_vec, 1));
        }
    }

    void copy_data_to_ptr(T *h_vec, int m_elem) const {
        if (m_elem != m_rows) {
            throw std::runtime_error("Vector: invalid m_elem dim for copy_data_to_ptr");
        }
        if (m_rows > 0) {
            check_cublas_status(cublasGetVector(m_rows, sizeof(T), d_vec, 1, h_vec, 1));
        }
    }

    // *** Conversion Constructors ***
    // Vector(const cublasHandle_t &arg_handle, const typename MatrixSparse<T>::Col &col):
    //     handle(arg_handle), m(col.rows()), mem_size(m*sizeof(T))
    // {

    //     allocate_d_vec();

    //     T *h_vec = static_cast<T *>(malloc(mem_size));
    //     for (int i=0; i<m; ++i) { h_vec[i] = col.coeff(i, 0); }
    //     cublasSetVector(m, sizeof(T), h_vec, 1, d_vec, 1);
    //     free(h_vec);

    // }

    // *** Destructor/Copy Constructor/Assignment Constructor ***
    virtual ~Vector() { check_cuda_error(cudaFree(d_vec)); }
    Vector<T> & operator=(const Vector<T> &other) {

        if (this != &other) {

            check_cuda_error(cudaFree(d_vec));

            handle = other.handle;
            m_rows = other.m_rows;
            mem_size = other.mem_size;

            allocate_d_vec();
            check_cuda_error(cudaMemcpy(d_vec, other.d_vec, mem_size, cudaMemcpyDeviceToDevice));

        }

        return *this;

    }
    Vector(const Vector<T> &other) {
        *this = other;
    }

    // *** Static Creation ***
    static Vector<T> Zero(const cublasHandle_t &arg_handle, int arg_m) {

        T *h_vec = static_cast<T *>(malloc(arg_m*sizeof(T)));

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<T>(0); }
        Vector<T> created_vec(arg_handle, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    static Vector<T> Zero(const cublasHandle_t &arg_handle, int arg_m, int arg_n) {
        check_n(arg_n);
        return Zero(arg_handle, arg_m);
    }

    static Vector<T> Ones(const cublasHandle_t &arg_handle, int arg_m) {

        T *h_vec = static_cast<T *>(malloc(arg_m*sizeof(T)));

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<T>(1); }
        Vector<T> created_vec(arg_handle, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    static Vector<T> Ones(const cublasHandle_t &arg_handle, int arg_m, int arg_n) {
        check_n(arg_n);
        return Ones(arg_handle, arg_m);
    }

    // Needed for testing (don't need to optimize performance)
    static Vector<T> Random(const cublasHandle_t &arg_handle, int arg_m) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        T *h_vec = static_cast<T *>(malloc(arg_m*sizeof(T)));

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<T>(dist(gen)); }
        Vector<T> created_vec(arg_handle, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    // Needed for testing (don't need to optimize performance)
    static Vector<T> Random(const cublasHandle_t &arg_handle, int arg_m, int arg_n) {
        check_n(arg_n);
        return Random(arg_handle, arg_m);
    }

    // *** Element Access ***
    const T get_elem(int row, int col) const {
        if (col != 0) {
            throw std::runtime_error("Vector: invalid vector col access in get_elem");
        }
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("Vector: invalid vector row access in get_elem");
        }
        T h_elem;
        check_cuda_error(cudaMemcpy(&h_elem, d_vec+row, sizeof(T), cudaMemcpyDeviceToHost));
        return h_elem;
    }
    const T get_elem(int row) const { return get_elem(row, 0); }
    void set_elem(int row, int col, T val) {
        if (col != 0) {
            throw std::runtime_error("Vector: invalid vector col access in set_elem");
        }
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("Vector: invalid vector row access in set_elem");
        }
        check_cuda_error(cudaMemcpy(d_vec+row, &val, sizeof(T), cudaMemcpyHostToDevice));
    }
    void set_elem(int row, T val) { set_elem(row, 0, val); }
    Vector<T> slice(int start, int m_elem) const {

        if ((m_elem < 0) || ((start+m_elem) > m_rows)) {
            throw(std::runtime_error("Vector: slice size invalid"));
        }
        if ((start < 0) || (start >= m_rows)) {
            throw(std::runtime_error("Vector: invalid slice start"));
        }

        T *h_vec = static_cast<T *>(malloc(m_elem*sizeof(T)));

        if (m_elem > 0) {
            check_cublas_status(cublasGetVector(m_elem, sizeof(T), d_vec+start, 1, h_vec, 1));
        }

        Vector<T> created_vec(handle, h_vec, m_elem);

        free(h_vec);

        return created_vec;

    }

    // *** Properties ***
    int rows() const { return m_rows; }
    int cols() const { return 1; }
    cublasHandle_t get_handle() const { return handle; }
    void print() const {

        T *h_vec = static_cast<T *>(malloc(mem_size));

        if (m_rows > 0) {
            check_cublas_status(cublasGetVector(m_rows, sizeof(T), d_vec, 1, h_vec, 1));
        }

        for (int i=0; i<m_rows; ++i) {
            std::cout << static_cast<double>(h_vec[i]) << std::endl;
        }
        std::cout << std::endl;

        free(h_vec);

    }

    // *** Resizing ***
    void reduce() { ; } // Do nothing since Vector is dense

    // *** Boolean ***
    bool operator==(const Vector<T> &other) const {
        
        if (this == &other) { return true; }
        if (m_rows != other.m_rows) { return false; }

        T *h_vec_self = static_cast<T *>(malloc(mem_size));
        T *h_vec_other = static_cast<T *>(malloc(mem_size));

        if (m_rows > 0) {
            check_cublas_status(cublasGetVector(m_rows, sizeof(T), d_vec, 1, h_vec_self, 1));
            check_cublas_status(cublasGetVector(m_rows, sizeof(T), other.d_vec, 1, h_vec_other, 1));
        }

        bool is_equal = true;
        for (int i=0; i<m_rows; ++i) { is_equal = is_equal && (h_vec_self[i] == h_vec_other[i]); }

        free(h_vec_self);
        free(h_vec_other);

        return is_equal;

    }

    // *** Explicit Cast ***
    template <typename Cast_T>
    Vector<Cast_T> cast() const {
        
        T *h_vec = static_cast<T *>(malloc(mem_size));
        Cast_T *h_cast_vec = static_cast<Cast_T *>(malloc(m_rows*sizeof(Cast_T)));

        if (m_rows > 0) {
            check_cublas_status(cublasGetVector(m_rows, sizeof(T), d_vec, 1, h_vec, 1));
        }

        for (int i=0; i<m_rows; ++i) { h_cast_vec[i] = static_cast<Cast_T>(h_vec[i]); }
        Vector<Cast_T> created_vec(handle, h_cast_vec, m_rows);

        free(h_vec);
        free(h_cast_vec);

        return created_vec;
    
    }

    template<>
    Vector<T> cast() const { return *this; } // Do nothing for same cast type

    // *** Arithmetic/Compound Operations ***
    Vector<T> operator*(const T &scalar) const;
    Vector<T> operator/(const T &scalar) const {
        return operator*(static_cast<T>(1.)/scalar);
    }

    Vector<T> & operator*=(const T &scalar);
    Vector<T> & operator/=(const T &scalar) {
        return operator*=(static_cast<T>(1.)/scalar);
    }

    Vector<T> operator+(const Vector<T> &vec) const;
    Vector<T> operator-(const Vector<T> &vec) const;

    Vector<T> & operator+=(const Vector<T> &vec);
    Vector<T> & operator-=(const Vector<T> &vec);
    
    T dot(const Vector<T> &vec) const;

    T norm() const;

    // *** Algorithms ***
    std::vector<int> sort_indices() const {
        
        int *h_indices = static_cast<int *>(malloc(m_rows*sizeof(int)));
        T *h_vec = static_cast<T *>(malloc(m_rows*sizeof(T)));

        for (int i=0; i<m_rows; ++i) { h_indices[i] = i; }
        cublasGetVector(m_rows, sizeof(T), d_vec, 1, h_vec, 1);

        vector_sort::quicksort(h_indices, h_vec, 0, m_rows);

        std::vector<int> indices_vec(m_rows);
        for (int i=0; i<m_rows; ++i) { indices_vec[i] = h_indices[i]; }

        free(h_indices);
        free(h_vec);

        return indices_vec;

    }

};

#include "MatrixDense.h"
#include "MatrixSparse.h"

#endif