#include "experiment_log.h"

void Experiment_Log::clear_file(fs::path file) {
    std::ofstream file_stream(file);
    file_stream.close();
}

Experiment_Log::Experiment_Log() {

    std::vector<spdlog::sink_ptr> logger_sinks;
    logger_sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    logger = std::make_shared<spdlog::logger>("", begin(logger_sinks), end(logger_sinks));
    logger->set_pattern("[%D %T] [%l] %v");

}

Experiment_Log::Experiment_Log(std::string logger_name, fs::path log_file, bool print_to_stdout) {

    clear_file(log_file);

    std::vector<spdlog::sink_ptr> logger_sinks;
    logger_sinks.push_back(
        std::make_shared<spdlog::sinks::basic_file_sink_st>(log_file.string())
    );
    if (print_to_stdout) {
        logger_sinks.push_back(
            std::make_shared<spdlog::sinks::stdout_sink_st>()
        );
    }

    logger = std::make_shared<spdlog::logger>(logger_name, begin(logger_sinks), end(logger_sinks));
    logger->set_pattern("[%D %T] [%l] %v");

}

void Experiment_Log::info(std::string s) {
    logger->info(s);
}

void Experiment_Log::warn(std::string s) {
    logger->warn(s);
}

void Experiment_Log::critical(std::string s) {
    logger->critical(s);
}