#include "types/Vector.h"
#include "types/Scalar.h"

#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using std::cout, std::endl;

int main() {

    // cublasHandle_t handle;
    // cublasCreate(&handle);
    
    // cublasPointerMode_t mode;
    // cublasGetPointerMode(handle, &mode);

    // cout << mode << endl; 
    // cublasLoggerConfigure(1, 1, 0, nullptr);
    // Vector<double> c(handle, {1, 2, 3, 4, 5});
    // Scalar<double> scalar(1.);
    // // (vec*scal).print();

    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    // c.print();

    // // cublasScalEx(
    // //     handle, 5,
    // //     &scalar.d_scalar, CUDA_R_64F,
    // //     c.d_vec, CUDA_R_64F, 1,
    // //     CUDA_R_64F
    // // );

    // cublasDaxpy(
    //     handle, 5,
    //     scalar.d_scalar,
    //     c.d_vec, 1,
    //     c.d_vec, 1
    // );

    // c.print();

    return 0;

}