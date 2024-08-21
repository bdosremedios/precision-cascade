#include "test_experiment.h"

#include "exp_tools/exp_tools.h"

#include <chrono>
#include <thread>

class Test_Experiment_Tools:
    public Test_Experiment_Base
{
public:

    void Test_Experiment_Clock_Basic_Run() {

        Experiment_Clock clock;

        const int delay_error = 15;

        const int first_elapse_1 = 10;

        ASSERT_FALSE(clock.check_completed());
        clock.start_clock_experiment();
        ASSERT_FALSE(clock.check_completed());
        std::this_thread::sleep_for(std::chrono::milliseconds(first_elapse_1));
        clock.stop_clock_experiment();
        ASSERT_TRUE(clock.check_completed());
        ASSERT_NEAR(clock.get_elapsed_time_ms(), first_elapse_1, delay_error);

        const int first_elapse_2 = 300;

        clock.start_clock_experiment();
        ASSERT_FALSE(clock.check_completed());
        std::this_thread::sleep_for(std::chrono::milliseconds(first_elapse_2));
        clock.stop_clock_experiment();
        ASSERT_TRUE(clock.check_completed());
        ASSERT_NEAR(clock.get_elapsed_time_ms(), first_elapse_2, delay_error);

        const int first_elapse_3 = 150;

        clock.start_clock_experiment();
        ASSERT_FALSE(clock.check_completed());
        std::this_thread::sleep_for(std::chrono::milliseconds(first_elapse_3));
        clock.stop_clock_experiment();
        ASSERT_TRUE(clock.check_completed());
        ASSERT_NEAR(clock.get_elapsed_time_ms(), first_elapse_3, delay_error);

    }

};

TEST_F(Test_Experiment_Tools, Test_Experiment_Clock_Basic_Run) {
    Test_Experiment_Clock_Basic_Run();
}