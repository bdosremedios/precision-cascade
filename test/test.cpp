#include "gtest/gtest.h"
#include <string>
#include <iostream>

using std::cout, std::endl;
using std::string;

int main(int argc, char **argv) {

    testing::InitGoogleTest();
    if ((argc > 1) && ((string(argv[1]) == "--run_long_tests") || (string(argv[1]) == "-rlt"))) {
        cout << "Running long tests..." << endl;
    } else {
        cout << "Skipping long tests..." << endl;
        testing::GTEST_FLAG(filter) = "-*LONGRUNTIME";
    }
    return RUN_ALL_TESTS();

}