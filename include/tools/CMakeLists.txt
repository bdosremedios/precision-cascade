cmake_minimum_required(VERSION 3.22)

project(gmres)

add_library(ArgPkg ArgPkg.cpp)

target_link_libraries(
    ArgPkg
    PRIVATE
    eigen
)

# target_include_directories(
#     ArgPkg
#     PRIVATE
#     "${CMAKE_CURRENT_SOURCE_DIR}/../include"
# )