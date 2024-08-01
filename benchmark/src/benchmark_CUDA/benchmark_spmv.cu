#include "benchmark.h"

#include "tools/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <random>
#include <vector>

class Benchmark_SPMV: public BenchmarkBase
{
public:

    double *d_one_dbl;
    float *d_one_sgl;
    __half *d_one_hlf;

    double *d_zero_dbl;
    float *d_zero_sgl;
    __half *d_zero_hlf;

    Benchmark_SPMV() {

        check_cuda_error(cudaMalloc(&d_one_dbl, sizeof(double)));
        check_cuda_error(cudaMalloc(&d_one_sgl, sizeof(float)));
        check_cuda_error(cudaMalloc(&d_one_hlf, sizeof(__half)));
        check_cuda_error(cudaMalloc(&d_zero_dbl, sizeof(double)));
        check_cuda_error(cudaMalloc(&d_zero_sgl, sizeof(float)));
        check_cuda_error(cudaMalloc(&d_zero_hlf, sizeof(__half)));
    

        double one_dbl = static_cast<double>(1.);
        float one_sgl = static_cast<float>(1.);
        __half one_hlf = static_cast<__half>(1.);

        check_cuda_error(cudaMemcpy(
            d_one_dbl, &one_dbl, sizeof(double), cudaMemcpyHostToDevice 
        ));
        check_cuda_error(cudaMemcpy(
            d_one_sgl, &one_sgl, sizeof(float), cudaMemcpyHostToDevice 
        ));
        check_cuda_error(cudaMemcpy(
            d_one_hlf, &one_hlf, sizeof(__half), cudaMemcpyHostToDevice 
        ));

        double zero_dbl = static_cast<double>(0.);
        float zero_sgl = static_cast<float>(0.);
        __half zero_hlf = static_cast<__half>(0.);

        check_cuda_error(cudaMemcpy(
            d_zero_dbl, &zero_dbl, sizeof(double),cudaMemcpyHostToDevice 
        ));
        check_cuda_error(cudaMemcpy(
            d_zero_sgl, &zero_sgl, sizeof(float),cudaMemcpyHostToDevice 
        ));
        check_cuda_error(cudaMemcpy(
            d_zero_hlf, &zero_hlf, sizeof(__half),cudaMemcpyHostToDevice 
        ));

    }

    ~Benchmark_SPMV() {

        check_cuda_error(cudaFree(d_one_dbl));
        check_cuda_error(cudaFree(d_one_sgl));
        check_cuda_error(cudaFree(d_one_hlf));
        
        check_cuda_error(cudaFree(d_zero_dbl));
        check_cuda_error(cudaFree(d_zero_sgl));
        check_cuda_error(cudaFree(d_zero_hlf));

    }

    template <typename TPrecision>
    void benchmark_spmv_func(
        std::vector<int> m_dimensions,
        std::function<void (Benchmark_AccumClock &, int, int, int *, int *, TPrecision *, TPrecision *, TPrecision *)> exec_func,
        std::string label
    ) {

        fs::path file_path = data_dir / fs::path(label + ".csv");
        std::ofstream f_out;
        f_out.open(file_path);

        if (!(f_out.is_open())) {
            throw std::runtime_error(
                "benchmark_spmv_func: " + file_path.string() + " did not open"
            );
        }

        f_out << "dim,med,avg,tot,min,max" << std::endl; 

        for (int m : m_dimensions) {

            Benchmark_AccumClock curr_clock;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> val_dist(0., 1.);
            std::uniform_real_distribution<double> fill_prob_dist(0., 1.);
            double fill_prob = std::sqrt(static_cast<double>(m))/static_cast<double>(m);

            TPrecision *vec_to_mult = static_cast<TPrecision *>(malloc(m*sizeof(TPrecision)));
            for (int k=0; k<m; ++k) { vec_to_mult[k] = val_dist(gen); }
            TPrecision *output = static_cast<TPrecision *>(malloc(m*sizeof(TPrecision)));

            TPrecision *d_vec_to_mult = nullptr;
            TPrecision *d_output = nullptr;

            check_cuda_error(cudaMalloc(&d_vec_to_mult, m*sizeof(TPrecision)));
            check_cuda_error(cudaMalloc(&d_output, m*sizeof(TPrecision)));

            check_cuda_error(cudaMemcpy(
                d_vec_to_mult,
                vec_to_mult,
                m*sizeof(TPrecision),
                cudaMemcpyHostToDevice 
            ));
            
            check_cuda_error(cudaMemcpy(
                d_output,
                output,
                m*sizeof(TPrecision),
                cudaMemcpyHostToDevice 
            ));

            free(vec_to_mult);
            free(output);

            int *offsets = static_cast<int *>(malloc((m+1)*sizeof(int)));
            std::vector<int> indices_vec;
            std::vector<TPrecision> vals_vec;

            int curr_nnz = 0;
            for (int j=0; j<m; ++j) {
                offsets[j] = curr_nnz;
                for (int i=0; i<m; ++i) {

                    TPrecision val = static_cast<TPrecision>(0.);

                    if (i == j) {
                        double temp = 0.1*val_dist(gen);
                        temp += temp/std::abs(temp);
                        val = static_cast<TPrecision>(temp);
                    } else if (
                        (fill_prob != 0.) &&
                        (fill_prob_dist(gen) <= fill_prob)
                    ) {
                        val = static_cast<TPrecision>(0.1*val_dist(gen));
                    }

                    if (val != static_cast<TPrecision>(0.)) {
                        indices_vec.push_back(i);
                        vals_vec.push_back(val);
                        ++curr_nnz;
                    }

                }
            }
            offsets[m] = curr_nnz;

            int *d_offsets = nullptr;
            int *d_indices = nullptr;
            TPrecision *d_vals = nullptr;
            
            check_cuda_error(cudaMalloc(&d_offsets, (m+1)*sizeof(int)));
            check_cuda_error(cudaMalloc(&d_indices, curr_nnz*sizeof(int)));
            check_cuda_error(cudaMalloc(&d_vals, curr_nnz*sizeof(TPrecision)));

            check_cuda_error(cudaMemcpy(
                d_offsets,
                offsets,
                (m+1)*sizeof(int),
                cudaMemcpyHostToDevice 
            ));
            check_cuda_error(cudaMemcpy(
                d_indices,
                &indices_vec[0],
                curr_nnz*sizeof(int),
                cudaMemcpyHostToDevice 
            ));
            check_cuda_error(cudaMemcpy(
                d_vals,
                &vals_vec[0],
                curr_nnz*sizeof(TPrecision),
                cudaMemcpyHostToDevice 
            ));

            free(offsets);

            std::function<void(Benchmark_AccumClock &)> test_func = (
                [exec_func, m, curr_nnz, d_offsets, d_indices, d_vals, d_vec_to_mult, d_output](
                    Benchmark_AccumClock &arg_clock
                ) {
                    exec_func(
                        arg_clock,
                        m, curr_nnz,
                        d_offsets, d_indices, d_vals,
                        d_vec_to_mult, d_output
                    );
                }
            );

            benchmark_n_runs(
                n_runs,
                test_func,
                curr_clock,
                label + "_" + std::to_string(m)
            );

            check_cuda_error(cudaFree(d_vec_to_mult));
            check_cuda_error(cudaFree(d_output));

            check_cuda_error(cudaFree(d_offsets));
            check_cuda_error(cudaFree(d_indices));
            check_cuda_error(cudaFree(d_vals));

            f_out << m << ","
                  << curr_clock.get_median().count() << ","
                  << curr_clock.get_avg().count() << ","
                  << curr_clock.get_total().count() << ","
                  << curr_clock.get_min().count() << ","
                  << curr_clock.get_max().count()
                  << std::endl; 

        }

        f_out.close();

    }

};

TEST_F(Benchmark_SPMV, Benchmark_CSC_MV_dbl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, double *, double *, double *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, double *d_vals,
        double *d_vec, double *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsc(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_64F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        double *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<double>(
        sparse_dims, execute_func, "csc_mv_dbl"
    );

}

TEST_F(Benchmark_SPMV, Benchmark_CSC_MV_sgl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, float *, float *, float *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, float *d_vals,
        float *d_vec, float *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsc(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_32F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<float>(
        sparse_dims, execute_func, "csc_mv_sgl"
    );

}


TEST_F(Benchmark_SPMV, Benchmark_CSC_MV_hlf) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, __half *, __half *, __half *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, __half *d_vals,
        __half *d_vec, __half *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsc(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_16F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<__half>(
        sparse_dims, execute_func, "csc_mv_hlf"
    );

}

TEST_F(Benchmark_SPMV, Benchmark_CSC_TMV_dbl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, double *, double *, double *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, double *d_vals,
        double *d_vec, double *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsc(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_64F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        double *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<double>(
        sparse_dims, execute_func, "csc_tmv_dbl"
    );

}

TEST_F(Benchmark_SPMV, Benchmark_CSC_TMV_sgl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, float *, float *, float *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, float *d_vals,
        float *d_vec, float *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsc(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_32F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<float>(
        sparse_dims, execute_func, "csc_tmv_sgl"
    );

}


TEST_F(Benchmark_SPMV, Benchmark_CSC_TMV_hlf) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, __half *, __half *, __half *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, __half *d_vals,
        __half *d_vec, __half *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsc(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_16F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<__half>(
        sparse_dims, execute_func, "csc_tmv_hlf"
    );

}

TEST_F(Benchmark_SPMV, Benchmark_CSR_MV_dbl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, double *, double *, double *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, double *d_vals,
        double *d_vec, double *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsr(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_64F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        double *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<double>(
        sparse_dims, execute_func, "csr_mv_dbl"
    );

}

TEST_F(Benchmark_SPMV, Benchmark_CSR_MV_sgl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, float *, float *, float *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, float *d_vals,
        float *d_vec, float *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsr(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_32F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<float>(
        sparse_dims, execute_func, "csr_mv_sgl"
    );

}


TEST_F(Benchmark_SPMV, Benchmark_CSR_MV_hlf) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, __half *, __half *, __half *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, __half *d_vals,
        __half *d_vec, __half *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsr(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_16F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<__half>(
        sparse_dims, execute_func, "csr_mv_hlf"
    );

}

TEST_F(Benchmark_SPMV, Benchmark_CSR_TMV_dbl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, double *, double *, double *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, double *d_vals,
        double *d_vec, double *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsr(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_64F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_64F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        double *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_dbl, spMatDescr, dnVecDescr_orig,
            d_zero_dbl, dnVecDescr_new,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<double>(
        sparse_dims, execute_func, "csr_tmv_dbl"
    );

}

TEST_F(Benchmark_SPMV, Benchmark_CSR_TMV_sgl) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, float *, float *, float *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, float *d_vals,
        float *d_vec, float *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsr(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_32F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_32F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<float>(
        sparse_dims, execute_func, "csr_tmv_sgl"
    );

}


TEST_F(Benchmark_SPMV, Benchmark_CSR_TMV_hlf) {

    std::function<void (Benchmark_AccumClock &, int, int, int *, int *, __half *, __half *, __half *)> execute_func = [this] (
        Benchmark_AccumClock &clock,
        int m, int nnz,
        int *d_offsets, int *d_indices, __half *d_vals,
        __half *d_vec, __half *d_output
    ) {

        clock.clock_start();
        
        for (int quick_k=0; quick_k<run_fast_tests_count; ++quick_k) {

        cusparseConstSpMatDescr_t spMatDescr;
        cusparseConstDnVecDescr_t dnVecDescr_orig;
        cusparseDnVecDescr_t dnVecDescr_new;
        
        check_cusparse_status(cusparseCreateConstCsr(
            &spMatDescr,
            m, m, nnz,
            d_offsets, d_indices, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateConstDnVec(
            &dnVecDescr_orig, m, d_vec, CUDA_R_16F
        ));
        check_cusparse_status(cusparseCreateDnVec(
            &dnVecDescr_new, m, d_output, CUDA_R_16F
        ));

        size_t bufferSize;
        check_cusparse_status(cusparseSpMV_bufferSize(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            &bufferSize
        ));

        float *d_buffer;
        check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

        check_cusparse_status(cusparseSpMV(
            BenchmarkBase::bundle.get_cusparse_handle(),
            CUSPARSE_OPERATION_TRANSPOSE,
            d_one_sgl, spMatDescr, dnVecDescr_orig,
            d_zero_sgl, dnVecDescr_new,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG1,
            d_buffer
        ));

        check_cuda_error(cudaFree(d_buffer));
        
        check_cusparse_status(cusparseDestroySpMat(spMatDescr));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
        check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

        }
 
        clock.clock_stop();

    };

    benchmark_spmv_func<__half>(
        sparse_dims, execute_func, "csr_tmv_hlf"
    );

}