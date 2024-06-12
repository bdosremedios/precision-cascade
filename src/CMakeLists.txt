cmake_minimum_required(VERSION 3.27)

project(precision-cascade)

find_package(CUDAToolkit)
enable_language(CUDA)

if(WIN32)
    set (CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(
    precision-cascade-gpu_kernels
    STATIC
    types/Scalar/Scalar_gpu_kernels.cu
    types/Vector/Vector_gpu_kernels.cu
    types/GeneralMatrix/GeneralMatrix_gpu_kernels.cu
    types/MatrixDense/MatrixDense_gpu_kernels.cu
    types/MatrixSparse/NoFillMatrixSparse_gpu_kernels.cu
)

if(WIN32)
    target_link_options(precision-cascade-gpu_kernels PUBLIC "/NODEFAULTLIB:LIBCMT")
endif()

target_link_libraries(
    precision-cascade-gpu_kernels
    PUBLIC
    precision-cascade-headers
)

if(WIN32)
    target_link_libraries(
        precision-cascade-gpu_kernels
        PUBLIC
        "$ENV{CUDA_PATH}/lib/x64/cuda.lib"
        "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
    )
elseif(UNIX)
    target_link_libraries(
        precision-cascade-gpu_kernels
        PUBLIC
        "/usr/local/cuda/lib64/libcudart.so"
        "/usr/local/cuda/lib64/libcublas.so"
    )
endif()

set_property(TARGET precision-cascade-gpu_kernels PROPERTY CUDA_ARCHITECTURES 80)

add_library(
    precision-cascade-constants
    STATIC
    types/Scalar/Scalar_constants.cpp
)

if(WIN32)
    target_link_options(precision-cascade-constants PUBLIC "/NODEFAULTLIB:LIBCMT")
endif()

target_link_libraries(
    precision-cascade-constants
    PUBLIC
    precision-cascade-headers
)

set_property(TARGET precision-cascade-constants PROPERTY CUDA_ARCHITECTURES 80)

add_library(
    precision-cascade-tools
    STATIC
    tools/cuda_check.cu
    tools/cuHandleBundle.cu
    tools/abs.cpp
)

target_link_libraries(
    precision-cascade-tools
    PUBLIC
    precision-cascade-headers
)

if(WIN32)
    target_link_libraries(
        precision-cascade-tools
        PUBLIC
        "$ENV{CUDA_PATH}/lib/x64/cuda.lib"
        "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
        "$ENV{CUDA_PATH}/lib/x64/cusparse.lib"
    )
elseif(UNIX)
    target_link_libraries(
        precision-cascade-tools
        PUBLIC
        "/usr/local/cuda/lib64/libcudart.so"
        "/usr/local/cuda/lib64/libcublas.so"
        "/usr/local/cuda/lib64/libcusparse.so"
    )
endif()

set_property(TARGET precision-cascade-tools PROPERTY CUDA_ARCHITECTURES 80)

add_library(
    precision-cascade
    STATIC
    types/Scalar/Scalar_hlf.cu
    types/Scalar/Scalar_sgl.cu
    types/Scalar/Scalar_dbl.cu
    types/Vector/Vector_hlf.cu
    types/Vector/Vector_sgl.cu
    types/Vector/Vector_dbl.cu
    types/MatrixDense/MatrixDense_general.cu
    types/MatrixDense/MatrixDense_hlf.cu
    types/MatrixDense/MatrixDense_sgl.cu
    types/MatrixDense/MatrixDense_dbl.cu
    types/MatrixSparse/NoFillMatrixSparse_general.cu
    types/MatrixSparse/NoFillMatrixSparse_hlf.cu
    types/MatrixSparse/NoFillMatrixSparse_sgl.cu
    types/MatrixSparse/NoFillMatrixSparse_dbl.cu
)

if(WIN32)
    target_link_options(precision-cascade PUBLIC "/NODEFAULTLIB:LIBCMT")
endif()

target_link_libraries(
    precision-cascade
    PUBLIC
    precision-cascade-headers
    precision-cascade-gpu_kernels
    precision-cascade-constants
    precision-cascade-tools
)

if(WIN32)
    target_link_libraries(
        precision-cascade
        PUBLIC
        "$ENV{CUDA_PATH}/lib/x64/cuda.lib"
        "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
        "$ENV{CUDA_PATH}/lib/x64/cusparse.lib"
    )
elseif(UNIX)
    target_link_libraries(
        precision-cascade
        PUBLIC
        "/usr/local/cuda/lib64/libcudart.so"
        "/usr/local/cuda/lib64/libcublas.so"
        "/usr/local/cuda/lib64/libcusparse.so"
    )
endif()

set_property(TARGET precision-cascade PROPERTY CUDA_ARCHITECTURES 80)