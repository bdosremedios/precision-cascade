#include "test_experiment.h"

#include <gtest/gtest.h>

#include <iostream>
#include <filesystem>

#ifdef WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

bool *Test_Experiment_Base::print_errors = new bool;
cuHandleBundle *Test_Experiment_Base::cu_handles_ptr = new cuHandleBundle();
fs::path Test_Experiment_Base::test_exp_data_dir;
fs::path Test_Experiment_Base::test_json_dir;
fs::path Test_Experiment_Base::test_data_dir;
fs::path Test_Experiment_Base::test_output_dir;

int main(int argc, char **argv) {

    #ifdef WIN32
        CHAR path[MAX_PATH];
        GetModuleFileNameA(NULL, path, MAX_PATH);
        Test_Experiment_Base::test_exp_data_dir = (
            fs::path(path).parent_path() /
            fs::path("data")
        );
    #else
        Test_Experiment_Base::test_exp_data_dir = (
            fs::canonical("/proc/self/exe").parent_path() /
            fs::path("data")
        );
    #endif
    Test_Experiment_Base::test_json_dir = (
        Test_Experiment_Base::test_exp_data_dir /
        fs::path("test_jsons")
    );
    Test_Experiment_Base::test_data_dir = (
        Test_Experiment_Base::test_exp_data_dir /
        fs::path("test_data")
    );
    Test_Experiment_Base::test_output_dir = (
        Test_Experiment_Base::test_exp_data_dir /
        fs::path("test_output")
    );

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
        *(Test_Experiment_Base::print_errors) = true;
    } else {
        std::cout << "Not printing expected errors..." << std::endl;
        *(Test_Experiment_Base::print_errors) = false;
    }

    Test_Experiment_Base::cu_handles_ptr->create();

    int return_status = RUN_ALL_TESTS();

    Test_Experiment_Base::cu_handles_ptr->destroy();

    delete Test_Experiment_Base::cu_handles_ptr;
    delete Test_Experiment_Base::print_errors;

    return return_status;

}