#include "test_experiment.h"

#include "exp_tools/exp_tools.h"

#include <chrono>
#include <thread>

class Test_Experiment_Tools:
    public Test_Experiment_Base
{
public:

    void Test_Experiment_Clock_Basic_Run() {

        const int time_ms = 150;
        Experiment_Clock clock;

        ASSERT_FALSE(clock.check_completed());
        clock.start_clock_experiment();
        ASSERT_FALSE(clock.check_completed());
        std::this_thread::sleep_for(std::chrono::milliseconds(time_ms));
        clock.stop_clock_experiment();
        ASSERT_TRUE(clock.check_completed());
        ASSERT_NEAR(clock.get_elapsed_time_ms(), time_ms, 1);

    }

};

TEST_F(Test_Experiment_Tools, Test_Experiment_Clock_Basic_Run) {
    Test_Experiment_Clock_Basic_Run();
}