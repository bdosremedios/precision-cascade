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

#include <cstdlib>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <vector>
#include <initializer_list>
#include <random>

template <typename TPrecision>
class Vector
{
private:

    static void check_n(int n) {
        if (n != 1) {
            throw std::runtime_error(
                "Vector: invalid number of columns for vector"
            );
        }
    }

    /* Allow all similar type Matrices and different type variants of self to
       access private methods */
    template <typename> friend class Vector;
    friend MatrixDense<TPrecision>;
    friend NoFillMatrixSparse<TPrecision>;

    cuHandleBundle cu_handles;
    int m_rows = 0;
    TPrecision *d_vec = nullptr;

    size_t mem_size() const {
        return m_rows*sizeof(TPrecision);
    }

    void allocate_d_vec() {
        check_cuda_error(cudaMalloc(&d_vec, mem_size()));
    }

    void check_vecvec_op_compatibility(const Vector<TPrecision> &other) const {
        if (m_rows != other.m_rows) {
            throw std::runtime_error(
                "Vector: incompatible vector sizes for vec-vec operation"
            );
        }
    }

    Vector<__half> to_half() const;
    Vector<float> to_float() const;
    Vector<double> to_double() const;

    // Use argument overload for type specification rather than explicit
    // specialization due to limitation in g++
    Vector<__half> cast(TypeIdentity<__half> __) const {
        return to_half();
    }
    Vector<float> cast(TypeIdentity<float> __) const {
        return to_float();
    }
    Vector<double> cast(TypeIdentity<double> __) const {
        return to_double();
    }

public:

    Vector(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n):
        cu_handles(arg_cu_handles), m_rows(arg_m)
    { 
        check_n(arg_n);
        allocate_d_vec();
    }

    Vector(const cuHandleBundle &arg_cu_handles, int arg_m):
        Vector(arg_cu_handles, arg_m, 1)
    {}
    Vector(const cuHandleBundle &arg_cu_handles):
        Vector(arg_cu_handles, 0)
    {}

    Vector(
        const cuHandleBundle &arg_cu_handles,
        std::initializer_list<TPrecision> li
    ):
        Vector(arg_cu_handles, li.size())
    {
        TPrecision *h_vec = static_cast<TPrecision *>(malloc(mem_size()));

        struct loop_vars {
            int i;
            typename std::initializer_list<TPrecision>::const_iterator curr;
        };
        for (
            loop_vars vars = {0, std::cbegin(li)};
            vars.curr != std::cend(li);
            ++vars.curr, ++vars.i
        ) {
            h_vec[vars.i] = *vars.curr;
        }
        if (m_rows > 0) {
            check_cublas_status(cublasSetVector(
                m_rows, sizeof(TPrecision), h_vec, 1, d_vec, 1
            ));
        }

        free(h_vec);
    }

    /* Dynamic Memory Constructor (assumes outer code handles dynamic memory
       properly) */
    Vector(
        const cuHandleBundle &arg_cu_handles,
        const TPrecision *h_vec,
        const int m_elem
    ):
        Vector(arg_cu_handles, m_elem)
    {
        if (m_elem > 0) {
            check_cublas_status(cublasSetVector(
                m_rows, sizeof(TPrecision), h_vec, 1, d_vec, 1
            ));
        }
    }

    Vector(
        const cuHandleBundle &arg_cu_handles,
        const TPrecision *h_vec,
        const int m_elem,
        const int n_elem
    ):
        Vector(arg_cu_handles, m_elem)
    {
        check_n(n_elem);
        if (m_elem > 0) {
            check_cublas_status(cublasSetVector(
                m_rows, sizeof(TPrecision), h_vec, 1, d_vec, 1
            ));
        }
    }

    void copy_data_to_ptr(TPrecision *h_vec, int m_elem) const {

        if (m_elem != m_rows) {
            throw std::runtime_error(
                "Vector: invalid m_elem dim for copy_data_to_ptr"
            );
        }

        if (m_rows > 0) {
            check_cublas_status(cublasGetVector(
                m_rows, sizeof(TPrecision), d_vec, 1, h_vec, 1
            ));
        }

    }

    virtual ~Vector() { check_cuda_error(cudaFree(d_vec)); }

    Vector<TPrecision> & operator=(const Vector<TPrecision> &other) {

        if (this != &other) {

            cu_handles = other.cu_handles;

            if (m_rows != other.m_rows) {
                check_cuda_error(cudaFree(d_vec));
                m_rows = other.m_rows;
                allocate_d_vec();
            }

            check_cuda_error(cudaMemcpy(
                d_vec, other.d_vec, mem_size(), cudaMemcpyDeviceToDevice
            ));

        }

        return *this;

    }

    Vector(const Vector<TPrecision> &other) { *this = other; }

    // Static Matrix Creation
    static Vector<TPrecision> Zero(
        const cuHandleBundle &arg_cu_handles, int arg_m
    ) {

        TPrecision *h_vec = static_cast<TPrecision *>(
            malloc(arg_m*sizeof(TPrecision))
        );

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<TPrecision>(0); }
        Vector<TPrecision> created_vec(arg_cu_handles, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    static Vector<TPrecision> Zero(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {
        check_n(arg_n);
        return Zero(arg_cu_handles, arg_m);
    }

    static Vector<TPrecision> Ones(
        const cuHandleBundle &arg_cu_handles, int arg_m
    ) {

        TPrecision *h_vec = static_cast<TPrecision *>(
            malloc(arg_m*sizeof(TPrecision))
        );

        for (int i=0; i<arg_m; ++i) { h_vec[i] = static_cast<TPrecision>(1); }
        Vector<TPrecision> created_vec(arg_cu_handles, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    static Vector<TPrecision> Ones(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {
        check_n(arg_n);
        return Ones(arg_cu_handles, arg_m);
    }

    // Needed for testing (don't need to optimize performance)
    static Vector<TPrecision> Random(
        const cuHandleBundle &arg_cu_handles, int arg_m
    ) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        TPrecision *h_vec = static_cast<TPrecision *>(
            malloc(arg_m*sizeof(TPrecision))
        );

        for (int i=0; i<arg_m; ++i) {
            h_vec[i] = static_cast<TPrecision>(dist(gen));
        }

        Vector<TPrecision> created_vec(arg_cu_handles, h_vec, arg_m);

        free(h_vec);

        return created_vec;

    }

    // Needed for testing (don't need to optimize performance)
    static Vector<TPrecision> Random(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {
        check_n(arg_n);
        return Random(arg_cu_handles, arg_m);
    }

    const Scalar<TPrecision> get_elem(int row, int col) const {

        if (col != 0) {
            throw std::runtime_error(
                "Vector: invalid vector col access in get_elem"
            );
        }
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error(
                "Vector: invalid vector row access in get_elem"
            );
        }

        Scalar<TPrecision> elem;
        check_cuda_error(cudaMemcpy(
            elem.d_scalar, d_vec+row, sizeof(TPrecision), cudaMemcpyDeviceToHost
        ));
        return elem;

    }

    const Scalar<TPrecision> get_elem(int row) const {
        return get_elem(row, 0);
    }

    void set_elem(int row, int col, const Scalar<TPrecision> &val) {

        if (col != 0) {
            throw std::runtime_error(
                "Vector: invalid vector col access in set_elem"
            );
        }
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error(
                "Vector: invalid vector row access in set_elem"
            );
        }

        check_cuda_error(cudaMemcpy(
            d_vec+row,
            val.d_scalar,
            sizeof(TPrecision),
            cudaMemcpyHostToDevice
        ));

    }

    void set_elem(int row, const Scalar<TPrecision> &val) {
        set_elem(row, 0, val);
    }

    Vector<TPrecision> get_slice(int start, int m_elem) const {

        if ((m_elem < 0) || ((start+m_elem) > m_rows)) {
            throw std::runtime_error("Vector: get_slice size invalid");
        }
        if ((start < 0) || (start >= m_rows)) {
            throw std::runtime_error("Vector: invalid get_slice start");
        }

        Vector<TPrecision> created_vec(cu_handles, m_elem);

        if (m_elem > 0) {
            check_cuda_error(cudaMemcpy(
                created_vec.d_vec,
                d_vec+start,
                m_elem*sizeof(TPrecision),
                cudaMemcpyDeviceToDevice
            ));
        }

        return created_vec;

    }

    void set_slice(int start, int m_elem, const Vector<TPrecision> &other) {

        if ((m_elem < 0) || ((start+m_elem) > m_rows)) {
            throw std::runtime_error(
                "Vector: set_slice size invalid"
            );
        }
        if ((start < 0) || (start >= m_rows)) {
            throw std::runtime_error(
                "Vector: invalid set_slice start"
            );
        }
        if (other.rows() != m_elem) {
            throw std::runtime_error(
                "Vector: set_slice given slice does not match m_elem"
            );
        }

        if (m_elem > 0) {
            check_cuda_error(cudaMemcpy(
                d_vec+start,
                other.d_vec,
                m_elem*sizeof(TPrecision),
                cudaMemcpyDeviceToDevice
            ));
        }

    }

    int rows() const { return m_rows; }
    int cols() const { return 1; }
    int non_zeros() const {
        
        TPrecision *h_vec = static_cast<TPrecision *>(malloc(mem_size()));
        copy_data_to_ptr(h_vec, m_rows);

        int nnz = 0;
        for (int i=0; i<m_rows; ++i) {
            if (h_vec[i] != static_cast<TPrecision>(0.))  {
                ++nnz;
            }
        }

        free(h_vec);

        return nnz;

    }
    cuHandleBundle get_cu_handles() const { return cu_handles; }
    std::string get_vector_string() const {

        TPrecision *h_vec = static_cast<TPrecision *>(malloc(mem_size()));

        copy_data_to_ptr(h_vec, m_rows);

        std::stringstream acc_strm;
        acc_strm << std::setprecision(8);
        for (int i=0; i<m_rows-1; ++i) {
            acc_strm << static_cast<double>(h_vec[i]) << "\n";
        }
        acc_strm << static_cast<double>(h_vec[m_rows-1]);

        free(h_vec);

        return acc_strm.str();
    
    }
    std::string get_info_string() const {
        int non_zeros_count = non_zeros();
        std::stringstream acc_strm;
        acc_strm << std::setprecision(3);
        acc_strm << "Rows: " << m_rows << " | "
                << "Non-zeroes: " << non_zeros_count << " | "
                << "Fill ratio: "
                << (static_cast<double>(non_zeros_count) /
                    static_cast<double>(rows()*cols())) << " | "
                << "Max magnitude: "
                << static_cast<double>(get_max_mag_elem().get_scalar());
        return acc_strm.str();
    }

    bool operator==(const Vector<TPrecision> &other) const {
        
        if (this == &other) { return true; }
        if (m_rows != other.m_rows) { return false; }

        TPrecision *h_vec_self = static_cast<TPrecision *>(malloc(mem_size()));
        TPrecision *h_vec_other = static_cast<TPrecision *>(malloc(mem_size()));

        if (m_rows > 0) {
            check_cublas_status(cublasGetVector(
                m_rows, sizeof(TPrecision),
                d_vec, 1, h_vec_self, 1)
            );
            check_cublas_status(cublasGetVector(
                m_rows, sizeof(TPrecision),
                other.d_vec, 1, h_vec_other, 1)
            );
        }

        bool is_equal = true;
        for (int i=0; i<m_rows; ++i) {
            is_equal = is_equal && (h_vec_self[i] == h_vec_other[i]);
        }

        free(h_vec_self);
        free(h_vec_other);

        return is_equal;

    }

    template <typename Cast_TPrecision>
    Vector<Cast_TPrecision> cast() const {
        return cast(TypeIdentity<Cast_TPrecision>());
    }

    Vector<TPrecision> operator*(const Scalar<TPrecision> &scalar) const;
    Vector<TPrecision> operator/(const Scalar<TPrecision> &scalar) const {
        Scalar<TPrecision> temp(scalar);
        return operator*(temp.reciprocol());
    }

    Vector<TPrecision> & operator*=(const Scalar<TPrecision> &scalar);
    Vector<TPrecision> & operator/=(const Scalar<TPrecision> &scalar) {
        Scalar<TPrecision> temp(scalar);
        return operator*=(temp.reciprocol());
    }

    Vector<TPrecision> operator+(const Vector<TPrecision> &vec) const;
    Vector<TPrecision> operator-(const Vector<TPrecision> &vec) const;

    Vector<TPrecision> & operator+=(const Vector<TPrecision> &vec);
    Vector<TPrecision> & operator-=(const Vector<TPrecision> &vec);
    
    Scalar<TPrecision> dot(const Vector<TPrecision> &vec) const;

    Scalar<TPrecision> norm() const;

    Scalar<TPrecision> get_max_mag_elem() const {
        
        TPrecision *h_vec = static_cast<TPrecision *>(malloc(mem_size()));
        copy_data_to_ptr(h_vec, m_rows);

        TPrecision max_mag = static_cast<TPrecision>(0.);
        for (int i=0; i<m_rows; ++i) {
            TPrecision temp = abs_ns::abs(h_vec[i]);
            if (temp > max_mag) {
                max_mag = temp;
            }
        }

        free(h_vec);

        return Scalar<TPrecision>(max_mag);

    }

};

#endif