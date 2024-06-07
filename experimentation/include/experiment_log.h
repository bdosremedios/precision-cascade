#ifndef EXPERIMENT_LOG_H
#define EXPERIMENT_LOG_H

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"

#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cstdio>

namespace fs = std::filesystem;

class Experiment_Log
{
private:

    void clear_file(fs::path file);

public:

    std::shared_ptr<spdlog::logger> logger;

    Experiment_Log();
    Experiment_Log(
        std::string logger_name, fs::path log_file, bool print_to_stdout
    );

    void info(std::string s);

    void warn(std::string s);

    void critical(std::string s);

};


#endif