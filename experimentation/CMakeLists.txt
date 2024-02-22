# cmake_minimum_required(VERSION 3.27.2)

# set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

# set (CMAKE_CXX_STANDARD 20)
# set (CMAKE_CUDA_STANDARD 20)
# set (CMAKE_CXX_STANDARD_REQUIRED ON)
# set (CMAKE_CUDA_STANDARD_REQUIRED ON)

# project(precision-cascade)

# add_library(experiment_tools experiment_tools.cpp)

# target_link_libraries(
#     experiment_tools
#     PUBLIC
#     precision-cascade
# )

# add_executable(experiment experiment.cpp)

# target_link_libraries(
#     experiment
#     PUBLIC
#     precision-cascade
#     experiment_tools
# )