#ifndef TEST_H
#define TEST_H

#include <cmath>
#include <filesystem>
#include <memory>
#include <iostream>

#include "gtest/gtest.h"

#include "types/types.h"

#include "test_assertions.h"
#include "test_tools.h"

namespace fs = std::filesystem;

class TestBase: public testing::Test
{
public:

    const fs::path read_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("read_matrices")
    );
    const fs::path solve_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("solve_matrices")
    );

    // SolveArgPkg default_args;
    static bool *show_plots;
    static cublasHandle_t *handle_ptr;
    static bool *print_errors;

};

#endif