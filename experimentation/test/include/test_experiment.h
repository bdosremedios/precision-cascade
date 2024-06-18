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
    static fs::path test_exp_data_dir;
    static fs::path test_json_dir;
    static fs::path test_data_dir;
    static fs::path test_output_dir;

};

#endif