cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

project(precision-cascade)

# add_executable(experiment experiment.cpp)

# target_link_libraries(
#     experiment
#     PUBLIC
#     eigen
#     precision-cascade
# )