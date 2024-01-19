#include "../../test.h"

#include "tools/arg_pkgs/SolveArgPkg.h"

class SolveArgPkg_Test: public TestBase {};

TEST_F(SolveArgPkg_Test, TestDefaultConstructionAndChecks) {

    SolveArgPkg args;

    ASSERT_TRUE(args.check_default_max_iter());
    ASSERT_TRUE(args.check_default_max_inner_iter());
    ASSERT_TRUE(args.check_default_target_rel_res());
    ASSERT_TRUE(args.check_default_init_guess());

    args.max_iter = 100;
    ASSERT_FALSE(args.check_default_max_iter());
    args.max_inner_iter = 10;
    ASSERT_FALSE(args.check_default_max_inner_iter());
    args.target_rel_res = 0.000001;
    ASSERT_FALSE(args.check_default_target_rel_res());
    args.init_guess = Vector<double>::Ones(*handle_ptr, 2, 1);
    ASSERT_FALSE(args.check_default_init_guess());

}

TEST_F(SolveArgPkg_Test, TestReset) {

    SolveArgPkg args;

    args.max_iter = 100;
    args.max_inner_iter = 10;
    args.target_rel_res = 0.000001;
    args.init_guess = Vector<double>::Ones(*handle_ptr, 2, 1);

    args = SolveArgPkg();

    ASSERT_TRUE(args.check_default_max_iter());
    ASSERT_TRUE(args.check_default_max_inner_iter());
    ASSERT_TRUE(args.check_default_target_rel_res());
    ASSERT_TRUE(args.check_default_init_guess());

}