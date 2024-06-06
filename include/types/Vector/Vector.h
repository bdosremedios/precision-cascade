#ifndef VECTOR_H
#define VECTOR_H

#include "tools/cuda_check.h"
#include "tools/cuHandleBundle.h"
#include "tools/abs.h"
#include "types/Scalar/Scalar.h"
#include "types/MatrixDense/MatrixDense.h"
#include "types/MatrixSparse/NoFillMatrixSparse.h"
#include "Vector_gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <random>
#include <initializer_list>

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
    friend NoFillMatrixSparse<T>;

    cuHandleBundle cu_handles;
    int m_rows = 0;
    T *d_vec = nullptr;

    size_t mem_size() const {
        return m_rows*sizeof(T);
    }

    void allocate_d_vec() {
        check_cuda_error(cudaMalloc(&d_vec, mem_size()));
    }

    void check_vecvec_op_compatibility(const Vector<T> &other) const {
        if (m_rows != other.m_rows) {
            throw std::runtime_error("Vector: incompatible vector sizes for vec-vec operation");
        }
    }

public:

    // *** Constructors ***
    Vector(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n):
        cu_handles(arg_cu_handles), m_rows(arg_m)
    { 
        check_n(arg_n);
        allocate_d_vec();
    }

    Vector(const cuHandleBundle &arg_cu_handles, int arg_m): Vector(arg_cu_handles, arg_m, 1) {}
    Vector(const cuHandleBundle &arg_cu_handles): Vector(arg_cu_handles, 0) {}

    Vector(const cuHandleBundle &arg_cu_handles, std::initializer_list<T> li):
        Vector(arg_cu_handles, li.size())
    {
        T *h_vec = static_cast<T *>(malloc(mem_size()));

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
    Vector(const cuHandleBundle &arg_cu_handles, const T *h_vec, const int m_elem):
        Vector(arg_cu_handles, m_elem)
    {
        if (m_elem > 0) {
            check_cublas_status(cublasSetVector(m_rows, sizeof(T), h_vec, 1, d_vec, 1));
        }
    }

    Vector(const cuHandleBundle &arg_cu_handles, const T *h_vec, const int m_elem, const int n_elem):
        Vector(arg_cu_handles, m_elem)
    {
        check_n(n_elem);
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
    Vector(const typename MatrixDense<T>::Col &col) {
        *this = col.copy_to_vec();
    }

    Vector(const typename NoFillMatrixSparse<T>::Col &col) {
        *this = col.copy_to_vec();
    }

    // *** Destructor ***
    virtual ~Vector() { check_cuda_error(cudaFree(d_vec)); }

    // *** Copy Assignment ***
    Vector<T> & operator=(const Vector<T> &other) {

        if (this != &other) {

            cu_handles = other.cu_handles;

            if (m_rows != other.m_rows) {
                check_cuda_error(cudaFree(d_vec));
                m_rows = other.m_rows;
                allocate_d_vec();
            }

            check_cuda_error(cudaMemcpy(d_vec, other.d_vec, mem_size(), cudaMemcpyDeviceToDevice));

        }

        return *this;

    }
    
    // *** Copy Constructor ***
    Vector(const Vector<T> &other) { *this = other; }

    // *** Static Creation ***
    static Vector<T> Zero(const cuHandleBundle &arg_cu_handles, int arg_m) {

        T *h_vec = static_cast<T *>(malloc(arg_m*sizeof(T)));

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<T>(0); }
        Vector<T> created_vec(arg_cu_handles, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    static Vector<T> Zero(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n) {
        check_n(arg_n);
        return Zero(arg_cu_handles, arg_m);
    }

    static Vector<T> Ones(const cuHandleBundle &arg_cu_handles, int arg_m) {

        T *h_vec = static_cast<T *>(malloc(arg_m*sizeof(T)));

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<T>(1); }
        Vector<T> created_vec(arg_cu_handles, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    static Vector<T> Ones(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n) {
        check_n(arg_n);
        return Ones(arg_cu_handles, arg_m);
    }

    // Needed for testing (don't need to optimize performance)
    static Vector<T> Random(const cuHandleBundle &arg_cu_handles, int arg_m) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        T *h_vec = static_cast<T *>(malloc(arg_m*sizeof(T)));

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<T>(dist(gen)); }
        Vector<T> created_vec(arg_cu_handles, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    // Needed for testing (don't need to optimize performance)
    static Vector<T> Random(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n) {
        check_n(arg_n);
        return Random(arg_cu_handles, arg_m);
    }

    // *** Element Access ***
    const Scalar<T> get_elem(int row, int col) const {

        if (col != 0) {
            throw std::runtime_error("Vector: invalid vector col access in get_elem");
        }
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("Vector: invalid vector row access in get_elem");
        }

        Scalar<T> elem;
        check_cuda_error(cudaMemcpy(elem.d_scalar, d_vec+row, sizeof(T), cudaMemcpyDeviceToHost));
        return elem;

    }

    const Scalar<T> get_elem(int row) const { return get_elem(row, 0); }

    void set_elem(int row, int col, const Scalar<T> &val) {

        if (col != 0) {
            throw std::runtime_error("Vector: invalid vector col access in set_elem");
        }
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("Vector: invalid vector row access in set_elem");
        }

        check_cuda_error(cudaMemcpy(d_vec+row, val.d_scalar, sizeof(T), cudaMemcpyHostToDevice));

    }

    void set_elem(int row, const Scalar<T> &val) { set_elem(row, 0, val); }

    Vector<T> get_slice(int start, int m_elem) const {

        if ((m_elem < 0) || ((start+m_elem) > m_rows)) {
            throw(std::runtime_error("Vector: get_slice size invalid"));
        }
        if ((start < 0) || (start >= m_rows)) {
            throw(std::runtime_error("Vector: invalid get_slice start"));
        }

        Vector<T> created_vec(cu_handles, m_elem);

        if (m_elem > 0) {
            check_cuda_error(cudaMemcpy(
                created_vec.d_vec, d_vec+start, m_elem*sizeof(T), cudaMemcpyDeviceToDevice
            ));
        }

        return created_vec;

    }

    void set_slice(int start, int m_elem, const Vector<T> &other) {

        if ((m_elem < 0) || ((start+m_elem) > m_rows)) {
            throw(std::runtime_error("Vector: set_slice size invalid"));
        }
        if ((start < 0) || (start >= m_rows)) {
            throw(std::runtime_error("Vector: invalid set_slice start"));
        }
        if (other.rows() != m_elem) {
            throw(std::runtime_error("Vector: set_slice given slice does not match m_elem"));
        }

        if (m_elem > 0) {
            check_cuda_error(cudaMemcpy(
                d_vec+start, other.d_vec, m_elem*sizeof(T), cudaMemcpyDeviceToDevice
            ));
        }

    }

    // *** Properties ***
    int rows() const { return m_rows; }
    int cols() const { return 1; }
    int non_zeros() const {
        
        T *h_vec = static_cast<T *>(malloc(mem_size()));
        copy_data_to_ptr(h_vec, m_rows);

        int nnz = 0;
        for (int i=0; i<m_rows; ++i) {
            if (h_vec[i] != static_cast<T>(0.))  {
                ++nnz;
            }
        }

        free(h_vec);

        return nnz;

    }
    cuHandleBundle get_cu_handles() const { return cu_handles; }
    std::string get_vector_string() const {

        T *h_vec = static_cast<T *>(malloc(mem_size()));

        copy_data_to_ptr(h_vec, m_rows);

        std::string acc;
        for (int i=0; i<m_rows-1; ++i) {
            acc += std::format("{:.8g}\n", static_cast<double>(h_vec[i]));
        }
        acc += std::format("{:.8g}", static_cast<double>(h_vec[m_rows-1]));

        free(h_vec);

        return acc;
    
    }
    std::string get_info_string() const {
        int non_zeros_count = non_zeros();
        return std::format(
            "Rows: {} | Non-zeroes: {} | Fill ratio: {:.3g} | Max magnitude: {:.3g}",
            m_rows,
            non_zeros_count,
            static_cast<double>(non_zeros_count)/static_cast<double>(rows()*cols()),
            static_cast<double>(get_max_mag_elem().get_scalar())
        );
    }

    // *** Boolean ***
    bool operator==(const Vector<T> &other) const {
        
        if (this == &other) { return true; }
        if (m_rows != other.m_rows) { return false; }

        T *h_vec_self = static_cast<T *>(malloc(mem_size()));
        T *h_vec_other = static_cast<T *>(malloc(mem_size()));

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

    // *** Cast ***
    template <typename Cast_T>
    Vector<Cast_T> cast() const { throw std::runtime_error("Vector: invalid cast conversion"); }

    Vector<__half> to_half() const;
    template <> Vector<__half> cast<__half>() const { return to_half(); }
    Vector<float> to_float() const;
    template <> Vector<float> cast<float>() const { return to_float(); }
    Vector<double> to_double() const;
    template <> Vector<double> cast<double>() const { return to_double(); }

    // *** Arithmetic/Compound Operations ***
    Vector<T> operator*(const Scalar<T> &scalar) const;
    Vector<T> operator/(const Scalar<T> &scalar) const {
        Scalar<T> temp(scalar);
        return operator*(temp.reciprocol());
    }

    Vector<T> & operator*=(const Scalar<T> &scalar);
    Vector<T> & operator/=(const Scalar<T> &scalar) {
        Scalar<T> temp(scalar);
        return operator*=(temp.reciprocol());
    }

    Vector<T> operator+(const Vector<T> &vec) const;
    Vector<T> operator-(const Vector<T> &vec) const;

    Vector<T> & operator+=(const Vector<T> &vec);
    Vector<T> & operator-=(const Vector<T> &vec);
    
    Scalar<T> dot(const Vector<T> &vec) const;

    Scalar<T> norm() const;

    Scalar<T> get_max_mag_elem() const {
        
        T *h_vec = static_cast<T *>(malloc(mem_size()));
        copy_data_to_ptr(h_vec, m_rows);

        T max_mag = static_cast<T>(0.);
        for (int i=0; i<m_rows; ++i) {
            T temp = abs_ns::abs(h_vec[i]);
            if (temp > max_mag) {
                max_mag = temp;
            }
        }

        free(h_vec);

        return Scalar<T>(max_mag);

    }

};

#endif