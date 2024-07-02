#ifndef PRECONDITIONER_DATA_H
#define PRECONDITIONER_DATA_H

#include "Timed_Experiment_Data.h"

#include "exp_tools/Experiment_Log.h"
#include "exp_spec/Preconditioner_Spec.h"

#include "tools/arg_pkgs/PrecondArgPkg.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

template <template <typename> typename TMatrix>
struct Preconditioner_Data:
    public Timed_Experiment_Data
{
private:

    void record_precond_data(std::ofstream &file_out) const {
        file_out << "\t\"id\" : \"" << id << "\",\n";
        file_out << "\t\"precond_left\" : \""
                 << typeid(*precond_arg_pkg_dbl.left_precond).name()
                 << "\",\n";
        file_out << "\t\"precond_right\" : \""
                 << typeid(*precond_arg_pkg_dbl.right_precond).name()
                 << "\",\n";
        file_out << "\t\"precond_specs\" : \""
                 << precond_specs.get_spec_string()
                 << "\"\n";
    }

public:

    Preconditioner_Spec precond_specs;
    cascade::PrecondArgPkg<TMatrix, double> precond_arg_pkg_dbl;

    Preconditioner_Data(
        std::string arg_precond_id,
        Experiment_Clock arg_clock,
        Preconditioner_Spec arg_precond_specs,
        cascade::PrecondArgPkg<TMatrix, double> arg_precond_arg_pkg_dbl
    ):
        Timed_Experiment_Data(arg_precond_id, arg_clock),
        precond_specs(arg_precond_specs),
        precond_arg_pkg_dbl(arg_precond_arg_pkg_dbl)
    {}

    std::string get_info_string() const override {
        return (
            clock.get_info_string() + " | " + precond_specs.get_spec_string()
        );
    }

    void record_json(
        std::string file_name, fs::path output_data_dir, Experiment_Log logger
    ) const override {

        std::ofstream file_out = open_json_ofstream(
            file_name, output_data_dir, logger
        );

        start_json(file_out);

        record_precond_data(file_out);

        end_json(file_out);

    }

};

#endif