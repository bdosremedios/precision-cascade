#ifndef TEST_H
#define TEST_H

#include "test_assertions.h"
#include "test_toolkit.h"

#include "tools/cuHandleBundle.h"
#include "tools/read_matrix.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "types/types.h"

#include "gtest/gtest.h"

#if defined(WIN32) && defined(_DEBUG)

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#endif

#include <memory>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

class TestBase: public testing::Test
{
public:

    #if defined(WIN32) && defined(_DEBUG)

    _CrtMemState init_state = {0};
    _CrtMemState final_state = {0};
    _CrtMemState state_diff = {0};

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
                std::cout << "LIKELY FALSE POSITIVE: 56 byte normal block leak "
                             "likely caused by GoogleTest"
                          << std::endl;
            } else {
                FAIL();
            }

        }

    }

    #endif

    static fs::path data_dir;
    static fs::path read_matrix_dir;
    static fs::path solve_matrix_dir;

    SolveArgPkg default_args;
    static bool *show_plots;
    static bool *print_errors;
    static cuHandleBundle bundle;

};

#endif