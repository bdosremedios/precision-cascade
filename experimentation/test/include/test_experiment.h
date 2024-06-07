#ifndef TEST_EXPERIMENT_H
#define TEST_EXPERIMENT_H

#include "test_assertions.h"

#include "tools/cuHandleBundle.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace fs = std::filesystem;

class TestExperimentBase: public testing::Test
{
public:

    static cuHandleBundle *cu_handles_ptr;
    static bool *print_errors;
    const fs::path test_exp_data_dir = (
        fs::current_path() / fs::path("..") / fs::path("experimentation") /
        fs::path("test") / fs::path("data")
    );
    const fs::path test_json_dir = (
        test_exp_data_dir / fs::path("test_jsons")
    );
    const fs::path test_data_dir = (
        test_exp_data_dir / fs::path("test_data")
    );
    const fs::path test_output_dir = (
        test_exp_data_dir / fs::path("test_output")
    );

};

#endif