cmake_minimum_required(VERSION 3.22)

project(gmres)

add_subdirectory(read_matrix)
add_subdirectory(solvers)
add_subdirectory(preconditioners)