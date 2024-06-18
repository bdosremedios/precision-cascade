#include <gtest/gtest.h>

#include "benchmark.h"

cuHandleBundle BenchmarkBase::bundle;
fs::path BenchmarkBase::data_dir;

int main() {
    
    #ifdef WIN32
        std::cout << fs::canonical("/proc/self/exe") << std::endl;
    #else
        BenchmarkBase::data_dir = (
            fs::canonical("/proc/self/exe").parent_path() /
            fs::path("data")
        );
    #endif

    testing::InitGoogleTest();

    BenchmarkBase::bundle.create();
    int return_status = RUN_ALL_TESTS();
    BenchmarkBase::bundle.destroy();

    return return_status;

}