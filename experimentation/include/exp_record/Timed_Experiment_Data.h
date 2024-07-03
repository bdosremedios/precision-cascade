#ifndef TIMED_EXPERIMENT_DATA_H
#define TIMED_EXPERIMENT_DATA_H

#include "exp_tools/Experiment_Clock.h"
#include "exp_tools/Experiment_Log.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

struct Timed_Experiment_Data
{
protected:
    
    std::ofstream open_json_ofstream(
        std::string file_name, fs::path save_dir, Experiment_Log logger
    ) const;

    void start_json(std::ofstream &file_out) const;

    void end_json(std::ofstream &file_out) const;

public:

    std::string id;
    Experiment_Clock clock;

    Timed_Experiment_Data(): id("") {}

    Timed_Experiment_Data(std::string arg_id, Experiment_Clock arg_clock):
        id(arg_id), clock(arg_clock)
    {}

    Timed_Experiment_Data(
        const Timed_Experiment_Data &other
    ) = default;

    Timed_Experiment_Data & operator=(
        const Timed_Experiment_Data &other
    ) = default;

    virtual std::string get_info_string() const = 0;
    virtual void record_json(
        std::string file_name, fs::path output_data_dir, Experiment_Log logger
    ) const = 0;

};

#endif