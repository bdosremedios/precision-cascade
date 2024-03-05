#include <iostream>

#include <gtest/gtest.h>

#include "test_experiment.h"

bool *TestExperimentBase::print_errors = new bool;

int main(int argc, char **argv) {

    testing::InitGoogleTest();

    // Check if should print errors
    bool print_errors = false;
    for (int i=0; i<argc; ++i) {
        if ((std::string(argv[i]) == "--print_errors") || (std::string(argv[i]) == "-pe")) {
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

    return RUN_ALL_TESTS();

}