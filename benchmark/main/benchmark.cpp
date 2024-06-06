#include <gtest/gtest.h>

#include "benchmark.h"

cuHandleBundle BenchmarkBase::bundle;

int main() {

    testing::InitGoogleTest();

    BenchmarkBase::bundle.create();
    int return_status = RUN_ALL_TESTS();
    BenchmarkBase::bundle.destroy();

    return return_status;

}