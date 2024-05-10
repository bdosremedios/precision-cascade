#include <functional>
#include <fstream>

#include <cmath>

#include "types/types.h"

#include "../test.h"

#include "include/benchmark_Sparse.h"

class NoFillMatrixSparse_Benchmark: public Benchmark_Sparse
{
public:

    // int min_n_mult = 8;
    // int max_n_mult = 15-prototying_n_speed_up;
    int min_n_mult = 11;
    int max_n_mult = 12;

    int min_n_substitution = 7;
    int max_n_substitution = 13-prototying_n_speed_up;

    std::function<NoFillMatrixSparse<double> (int, int)> make_low_tri_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return make_lower_tri(make_A_dbl(m, n));
    };
    
    std::function<NoFillMatrixSparse<double> (int, int)> make_upp_tri_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return make_upper_tri(make_A_dbl(m, n));
    };

};

TEST_F(NoFillMatrixSparse_Benchmark, MatrixVectorMult_Double_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
        NoFillMatrixSparse<double> &A, Vector<double> &b
    ) {
        A*b;
    };

    basic_func_benchmark<NoFillMatrixSparse, double>(
        min_n_mult, max_n_mult, make_A_dbl, execute_func, "matsparse_mv_dbl"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, MatrixVectorMult_Single_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<float> &, Vector<float> &)> execute_func = [] (
        NoFillMatrixSparse<float> &A, Vector<float> &b
    ) {
        A*b;
    };

    basic_func_benchmark<NoFillMatrixSparse, float>(
        min_n_mult, max_n_mult, make_A_sgl, execute_func, "matsparse_mv_sgl"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, MatrixVectorMult_Half_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<__half> &, Vector<__half> &)> execute_func = [] (
        NoFillMatrixSparse<__half> &A, Vector<__half> &b
    ) {
        std::cout << (A*(A*b)).get_info_string() << std::endl;
    };

    basic_func_benchmark<NoFillMatrixSparse, __half>(
        min_n_mult, max_n_mult, make_A_hlf, execute_func, "matsparse_mv_hlf"
    );

}

// TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixVectorMult_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
//         NoFillMatrixSparse<double> &A, Vector<double> &b
//     ) {
//         A.transpose_prod(b);
//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         min_n_mult, max_n_mult, make_A, execute_func, "matsparse_tmv"
//     );

// }

// TEST_F(NoFillMatrixSparse_Benchmark, MatrixBlockVectorMult_BENCHMARK) {
// }

// TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixBlockVectorMult_BENCHMARK) {
// }

// TEST_F(NoFillMatrixSparse_Benchmark, ForwardSubstitution_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
//         NoFillMatrixSparse<double> &A, Vector<double> &b
//     ) {
//         A.frwd_sub(b);
//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         min_n_substitution, max_n_substitution, make_low_tri_A, execute_func, "matsparse_frwdsub"
//     );

// }

// TEST_F(NoFillMatrixSparse_Benchmark, BackwardSubstitution_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
//         NoFillMatrixSparse<double> &A, Vector<double> &b
//     ) {
//         A.back_sub(b);
//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         min_n_substitution, max_n_substitution, make_upp_tri_A, execute_func, "matsparse_backsub"
//     );

// }