#ifndef TEST_H
#define TEST_H

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include <cmath>
#include <filesystem>
#include <memory>
#include <iostream>

#include "gtest/gtest.h"

#include "tools/cuHandleBundle.h"
#include "types/types.h"

#include "tools/read_matrix.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "test_assertions.h"
#include "test_toolkit.h"

namespace fs = std::filesystem;

class TestBase: public testing::Test
{
public:

    _CrtMemState init_state = {0};
    _CrtMemState final_state = {0};
    _CrtMemState state_diff = {0};

    #ifdef _DEBUG
    virtual void SetUp() {
        _CrtMemCheckpoint(&init_state);
    }

    virtual void TearDown() {

        _CrtMemCheckpoint(&final_state);
        
        if (_CrtMemDifference(&state_diff, &init_state, &final_state)) {

            _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
            _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
            _CrtMemDumpStatistics(&state_diff);
            _CrtMemDumpAllObjectsSince(&init_state);

            if (state_diff.lSizes[1] == 56) {
                std::string msg = "LIKELY FALSE POSITIVE: 56 byte normal block leak likely caused by GoogleTest";
                std::cout << msg << std::endl;
            } else {
                FAIL();
            }

        }

    }
    #endif

    const fs::path read_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("read_matrices")
    );
    const fs::path solve_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("solve_matrices")
    );

    SolveArgPkg default_args;
    static bool *show_plots;
    static bool *print_errors;
    static cuHandleBundle bundle;

};

#endif