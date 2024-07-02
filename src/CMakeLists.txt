cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_library(
    precision-cascade-gpu-kernels
    STATIC
    types/Scalar/Scalar_gpu_kernels.cu
    types/Vector/Vector_gpu_kernels.cu
    types/GeneralMatrix/GeneralMatrix_gpu_kernels.cu
    types/MatrixDense/MatrixDense_gpu_kernels.cu
    types/MatrixSparse/NoFillMatrixSparse_gpu_kernels.cu
)

if(WIN32)
    target_link_options(
        precision-cascade-gpu-kernels
        PUBLIC
        "/NODEFAULTLIB:LIBCMT"
    )
endif()

target_link_libraries(
    precision-cascade-gpu-kernels
    PUBLIC
    precision-cascade-headers
)

set_property(
    TARGET precision-cascade-gpu-kernels
    PROPERTY CUDA_ARCHITECTURES
    ${PROJECT_CUDA_ARCH_REAL} ${PROJECT_CUDA_ARCH_VRTL}
)

add_library(
    precision-cascade-constants
    STATIC
    types/Scalar/Scalar_constants.cpp
)

if(WIN32)
    target_link_options(
        precision-cascade-constants
        PUBLIC
        "/NODEFAULTLIB:LIBCMT"
    )
endif()

target_link_libraries(
    precision-cascade-constants
    PUBLIC
    precision-cascade-headers
)

set_property(
    TARGET precision-cascade-constants
    PROPERTY CUDA_ARCHITECTURES
    ${PROJECT_CUDA_ARCH_REAL} ${PROJECT_CUDA_ARCH_VRTL}
)

add_library(
    precision-cascade-tools
    STATIC
    tools/cuda_check.cu
    tools/cuHandleBundle.cu
    tools/read_matrix.cpp
    tools/abs.cpp
)

target_link_libraries(
    precision-cascade-tools
    PUBLIC
    precision-cascade-headers
)

set_property(
    TARGET precision-cascade-tools
    PROPERTY CUDA_ARCHITECTURES
    ${PROJECT_CUDA_ARCH_REAL} ${PROJECT_CUDA_ARCH_VRTL}
)

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
    target_link_options(
        precision-cascade
        PUBLIC
        "/NODEFAULTLIB:LIBCMT"
    )
endif()

target_link_libraries(
    precision-cascade
    PUBLIC
    precision-cascade-headers
    precision-cascade-gpu-kernels
    precision-cascade-constants
    precision-cascade-tools
)

set_property(
    TARGET precision-cascade
    PROPERTY CUDA_ARCHITECTURES
    ${PROJECT_CUDA_ARCH_REAL} ${PROJECT_CUDA_ARCH_VRTL}
)