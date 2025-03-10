#ifndef SOLVE_DATA_H
#define SOLVE_DATA_H

#include "Timed_Experiment_Data.h"

#include "exp_tools/write_json.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/VP_GMRES_IR.h"

template <
    template <template <typename> typename> typename TSolver,
    template <typename> typename TMatrix
>
struct Solve_Data:
    public Timed_Experiment_Data
{
private:

    void record_basic_solver_data(std::ofstream &file_out) const {
        file_out << "\t\"id\" : \"" << id << "\",\n";
        file_out << "\t\"solver_class\" : \""
                 << typeid(*solver_ptr).name()
                 << "\",\n";
        file_out << "\t\"initiated\" : \""
                 << write_json::bool_to_string(solver_ptr->check_initiated())
                 << "\",\n";
        file_out << "\t\"converged\" : \""
                 << write_json::bool_to_string(solver_ptr->check_converged())
                 << "\",\n";
        file_out << "\t\"terminated\" : \""
                 << write_json::bool_to_string(solver_ptr->check_terminated())
                 << "\",\n";
        file_out << "\t\"outer_iterations\" : "
                 << solver_ptr->get_iteration()
                 << ",\n";
        file_out << "\t\"inner_iterations\" : "
                 << write_json::int_vector_to_jsonarray_str(
                        solver_ptr->get_inner_iterations(), 0
                    )
                 << ",\n";
        file_out << "\t\"elapsed_time_ms\" : " << clock.get_elapsed_time_ms()
                 << ",\n";
    }

    void record_residual_solver_data(std::ofstream &file_out) const {

        file_out << "\t\"outer_res_norm_history\" : "
                 << write_json::dbl_vector_to_jsonarray_str(
                        solver_ptr->get_res_norm_history(), 0
                    )
                 << ",\n";

        file_out << "\t\"inner_res_norm_history\" : [\n";

        std::vector<std::vector<double>> inner_res_norm_history(
            solver_ptr->get_inner_res_norm_history()
        );
        for (int i=0; i<inner_res_norm_history.size()-1; ++i) {
            file_out << write_json::dbl_vector_to_jsonarray_str(
                            inner_res_norm_history[i],
                            2
                        )
                     << ",\n";
        }
        file_out << write_json::dbl_vector_to_jsonarray_str(
                        inner_res_norm_history[inner_res_norm_history.size()-1],
                        2
                    )
                 << "\n";

        file_out << "\t]\n";

    }

    void record_solver_data(
        std::ofstream &file_out,
        const std::shared_ptr<cascade::InnerOuterSolve<TMatrix>> &solver_ptr
    ) const {
        record_basic_solver_data(file_out);
        record_residual_solver_data(file_out);
    }

    void record_solver_data(
        std::ofstream &file_out,
        const std::shared_ptr<cascade::VP_GMRES_IR_Solve<TMatrix>> &solver_ptr
    ) const {
        record_basic_solver_data(file_out);
        file_out << "\t\"hlf_sgl_cascade_change\" : "
                 << solver_ptr->get_hlf_sgl_cascade_change()
                 << ",\n";
        file_out << "\t\"sgl_dbl_cascade_change\" : "
                 << solver_ptr->get_sgl_dbl_cascade_change()
                 << ",\n";
        record_residual_solver_data(file_out);
    }

public:

    std::shared_ptr<TSolver<TMatrix>> solver_ptr;

    Solve_Data(): Timed_Experiment_Data(), solver_ptr(nullptr) {}

    Solve_Data(
        std::string arg_solver_id,
        Experiment_Clock arg_clock,
        std::shared_ptr<TSolver<TMatrix>> arg_solver_ptr
    ):
        Timed_Experiment_Data(arg_solver_id, arg_clock),
        solver_ptr(arg_solver_ptr) 
    {}

    ~Solve_Data() = default;

    Solve_Data(const Solve_Data &other) {
        *this = other;
    }

    void operator=(const Solve_Data &other) {
        id = other.id;
        clock = other.clock;
        solver_ptr = other.solver_ptr;
    }

    std::string get_info_string() const override {
        return clock.get_info_string() + " | " + solver_ptr->get_info_string();
    }

    void record_json(
        std::string file_name,
        fs::path output_data_dir,
        Experiment_Log logger
    ) const override {

        std::ofstream file_out = write_json::open_json_ofstream(
            file_name, output_data_dir, logger
        );

        write_json::start_json(file_out);

        record_solver_data(file_out, solver_ptr);

        write_json::end_json(file_out);

    }

};

#endif