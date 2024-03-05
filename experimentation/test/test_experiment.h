#ifndef TEST_EXPERIMENT_H
#define TEST_EXPERIMENT_H

#include <filesystem>

#include "test_assertions.h"

namespace fs = std::filesystem;

class TestExperimentBase: public testing::Test
{
public:

    static bool *print_errors;
    const fs::path test_json_dir = (
        fs::current_path() /
        fs::path("..") /
        fs::path("experimentation") /
        fs::path("test") /
        fs::path("test_jsons")
    );

};

#endif