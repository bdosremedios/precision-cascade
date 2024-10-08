#include <gtest/gtest.h>

#include "benchmark.h"

#ifdef WIN32
#include <windows.h>
#endif

cuHandleBundle BenchmarkBase::bundle;
fs::path BenchmarkBase::data_dir;
bool BenchmarkBase::prototyping_speed_up = false;

int main(int argc, char *argv[]) {

    if (
        (argc == 2) &&
        ((std::string(argv[1]) == "--help") ||
         (std::string(argv[1]) == "-h"))
    ) {
        std::cout << "---- Entry point of precision-cascade benchmarking ----"
                  << std::endl;
        std::cout << "REQUIRES: \"output_data\" directory in same "
                     "directory as executable"
                  << std::endl;
        std::cout << "- \"output_data\" is the directory to store output "
                     "benchmarking data in"
                  << std::endl;
        std::cout << "option --prototype or -p runs quick versions of tests "
                     "for mocking purposes"
                  << std::endl;
        return EXIT_SUCCESS;
    } else if (
        (argc == 2) &&
        ((std::string(argv[1]) == "--prototype") ||
         (std::string(argv[1]) == "-p"))
    ) {
        std::cout << "Running with fast prototyping settings..."
                  << std::endl;
        BenchmarkBase::prototyping_speed_up = true;
    } else if (argc == 1) {
        ;
    } else {
        std::cout << "Invalid argument must be --help or --prototype or nothing"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Assumes output_data is in the same directory as executable
    #ifdef WIN32
        CHAR path[MAX_PATH];
        GetModuleFileNameA(NULL, path, MAX_PATH);
        BenchmarkBase::data_dir = (
            fs::path(path).parent_path() / fs::path("output_data")
        );
    #else
        BenchmarkBase::data_dir = (
            fs::canonical("/proc/self/exe").parent_path() /
            fs::path("output_data")
        );
    #endif

    // Check existence of directories
    if (!fs::is_directory(BenchmarkBase::data_dir)) {
        std::cerr << "Invalid benchmark output_data directory"
                  << std::endl;
        return EXIT_FAILURE;
    }

    testing::InitGoogleTest();

    BenchmarkBase::bundle.create();
    int return_status = RUN_ALL_TESTS();
    BenchmarkBase::bundle.destroy();

    return return_status;

}