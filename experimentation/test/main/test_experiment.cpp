#include "test_experiment.h"

#include <gtest/gtest.h>

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

bool *TestExperimentBase::print_errors = new bool;
cuHandleBundle *TestExperimentBase::cu_handles_ptr = new cuHandleBundle();
fs::path TestExperimentBase::test_exp_data_dir;
fs::path TestExperimentBase::test_json_dir;
fs::path TestExperimentBase::test_data_dir;
fs::path TestExperimentBase::test_output_dir;

int main(int argc, char **argv) {

    #ifdef WIN32
        std::cout << fs::canonical("/proc/self/exe") << std::endl;
    #else
        TestExperimentBase::test_exp_data_dir = (
            fs::canonical("/proc/self/exe").parent_path() /
            fs::path("data")
        );
        TestExperimentBase::test_json_dir = (
            TestExperimentBase::test_exp_data_dir /
            fs::path("test_jsons")
        );
        TestExperimentBase::test_data_dir = (
            TestExperimentBase::test_exp_data_dir /
            fs::path("test_data")
        );
        TestExperimentBase::test_output_dir = (
            TestExperimentBase::test_exp_data_dir /
            fs::path("test_output")
        );
    #endif

    testing::InitGoogleTest();

    // Check if should print errors
    bool print_errors = false;
    for (int i=0; i<argc; ++i) {
        if (
            (std::string(argv[i]) == "--print_errors") ||
            (std::string(argv[i]) == "-pe")
        ) {
            print_errors = true;
        }
    }
    if (print_errors) {
        std::cout << "Printing expected errors..." << std::endl;
        *(TestExperimentBase::print_errors) = true;
    } else {
        std::cout << "Not printing expected errors..." << std::endl;
        *(TestExperimentBase::print_errors) = false;
    }

    TestExperimentBase::cu_handles_ptr->create();

    int return_status = RUN_ALL_TESTS();

    TestExperimentBase::cu_handles_ptr->destroy();

    delete TestExperimentBase::cu_handles_ptr;
    delete TestExperimentBase::print_errors;

    return return_status;

}