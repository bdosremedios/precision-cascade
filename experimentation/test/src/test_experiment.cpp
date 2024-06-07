#include "test_experiment.h"

#include <gtest/gtest.h>

#include <iostream>

bool *TestExperimentBase::print_errors = new bool;
cuHandleBundle *TestExperimentBase::cu_handles_ptr = new cuHandleBundle();

int main(int argc, char **argv) {

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