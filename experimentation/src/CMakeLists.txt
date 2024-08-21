cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

set(
    EXPERIMENT_RUN_CPPS
    exp_run/exp_run_record.cpp
)

set(
    EXPERIMENT_READ_CPPS
    exp_read/exp_read.cpp
)

set(
    EXPERIMENT_SPECIFICATION_CPPS
    exp_spec/Preconditioner_Spec.cpp
    exp_spec/Solve_Group.cpp
    exp_spec/Experiment_Spec.cpp
)

set(
    EXPERIMENT_TOOLS_CPPS
    exp_tools/Experiment_Clock.cpp
    exp_tools/Experiment_Log.cpp
    exp_tools/write_json.cpp
)

add_library(
    experiment-lib
    ${EXPERIMENT_READ_CPPS}
    ${EXPERIMENT_RUN_CPPS}
    ${EXPERIMENT_SPECIFICATION_CPPS}
    ${EXPERIMENT_TOOLS_CPPS}
)

target_link_libraries(
    experiment-lib
    PUBLIC
    experiment-headers
)