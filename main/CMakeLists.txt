cmake_minimum_required(VERSION 3.22)

project(gmres)

add_executable(main main.cpp)

target_link_libraries(main eigen ArgPkg)

target_include_directories(
    main
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/"
)