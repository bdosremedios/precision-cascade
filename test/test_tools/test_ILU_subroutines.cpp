#include "../test.h"

#include "tools/ILU_subroutines.h"

// using namespace ilu;

// class ColVal_DblLink_Test: public TestBase
// {
// public:

//     template <typename T>
//     void TestConstruction() {

//         ColValHeadManager<T> manager(3);

//         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
//         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 1, static_cast<T>(0.), &manager);
//         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 2, static_cast<T>(10.2), &manager);

//         ASSERT_EQ(val_node_ptr_1->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_2->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_3->val, static_cast<T>(10.2));

//         ASSERT_EQ(val_node_ptr_1->row, 0);
//         ASSERT_EQ(val_node_ptr_2->row, 1);
//         ASSERT_EQ(val_node_ptr_3->row, 2);

//         ASSERT_EQ(val_node_ptr_1->col, 0);
//         ASSERT_EQ(val_node_ptr_2->col, 1);
//         ASSERT_EQ(val_node_ptr_3->col, 2);

//         delete val_node_ptr_1;
//         delete val_node_ptr_2;
//         delete val_node_ptr_3;

//     }

//     template <typename T>
//     void TestChaining() {

//         ColValHeadManager<T> manager(3);

//         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
//         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 1, static_cast<T>(0.), &manager);
//         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 2, static_cast<T>(10.2), &manager);

//         val_node_ptr_1->connect(val_node_ptr_2);
//         val_node_ptr_2->connect(val_node_ptr_3);

//         ASSERT_EQ(val_node_ptr_1->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->val, static_cast<T>(10.2));

//         ASSERT_EQ(val_node_ptr_3->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_3->prev_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_3->prev_col_val->prev_col_val->val, static_cast<T>(-1.5));

//         delete val_node_ptr_1;
//         delete val_node_ptr_2;
//         delete val_node_ptr_3;

//     }

//     template <typename T>
//     void DeleteMiddleofChain() {

//         ColValHeadManager<T> manager(3);

//         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
//         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 0, static_cast<T>(0.), &manager);
//         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 0, static_cast<T>(10.2), &manager);
//         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(3, 0, static_cast<T>(-0.5), &manager);

//         val_node_ptr_1->connect(val_node_ptr_2);
//         val_node_ptr_2->connect(val_node_ptr_3);
//         val_node_ptr_3->connect(val_node_ptr_4);

//         ASSERT_EQ(val_node_ptr_1->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->next_col_val->val, static_cast<T>(-0.5));

//         delete val_node_ptr_2;

//         ASSERT_EQ(val_node_ptr_1->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->val, static_cast<T>(-0.5));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->next_col_val, nullptr);

//         ASSERT_EQ(val_node_ptr_4->val, static_cast<T>(-0.5));
//         ASSERT_EQ(val_node_ptr_4->prev_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_4->prev_col_val->prev_col_val->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_4->prev_col_val->prev_col_val->prev_col_val, nullptr);
        
//         delete val_node_ptr_1;
//         delete val_node_ptr_3;
//         delete val_node_ptr_4;

//     }

//     template <typename T>
//     void DeleteTopofChain() {

//         ColValHeadManager<T> manager(3);

//         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 1, static_cast<T>(-1.5), &manager);
//         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 1, static_cast<T>(0.), &manager);
//         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 1, static_cast<T>(10.2), &manager);
//         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(3, 1, static_cast<T>(-0.5), &manager);

//         val_node_ptr_1->connect(val_node_ptr_2);
//         val_node_ptr_2->connect(val_node_ptr_3);
//         val_node_ptr_3->connect(val_node_ptr_4);

//         ASSERT_EQ(val_node_ptr_1->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->next_col_val->val, static_cast<T>(-0.5));

//         delete val_node_ptr_1;

//         ASSERT_EQ(val_node_ptr_2->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_2->next_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_2->next_col_val->next_col_val->val, static_cast<T>(-0.5));
//         ASSERT_EQ(val_node_ptr_2->next_col_val->next_col_val->next_col_val, nullptr);

//         ASSERT_EQ(val_node_ptr_4->val, static_cast<T>(-0.5));
//         ASSERT_EQ(val_node_ptr_4->prev_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_4->prev_col_val->prev_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_4->prev_col_val->prev_col_val->prev_col_val, nullptr);
        
//         delete val_node_ptr_2;
//         delete val_node_ptr_3;
//         delete val_node_ptr_4;

//     }

//     template <typename T>
//     void DeleteBottomofChain() {

//         ColValHeadManager<T> manager(3);

//         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 1, static_cast<T>(-1.5), &manager);
//         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 1, static_cast<T>(0.), &manager);
//         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 1, static_cast<T>(10.2), &manager);
//         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(3, 1, static_cast<T>(-0.5), &manager);

//         val_node_ptr_1->connect(val_node_ptr_2);
//         val_node_ptr_2->connect(val_node_ptr_3);
//         val_node_ptr_3->connect(val_node_ptr_4);

//         ASSERT_EQ(val_node_ptr_1->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->next_col_val->val, static_cast<T>(-0.5));

//         delete val_node_ptr_4;

//         ASSERT_EQ(val_node_ptr_1->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_1->next_col_val->next_col_val->next_col_val, nullptr);

//         ASSERT_EQ(val_node_ptr_3->val, static_cast<T>(10.2));
//         ASSERT_EQ(val_node_ptr_3->prev_col_val->val, static_cast<T>(0.));
//         ASSERT_EQ(val_node_ptr_3->prev_col_val->prev_col_val->val, static_cast<T>(-1.5));
//         ASSERT_EQ(val_node_ptr_3->prev_col_val->prev_col_val->prev_col_val, nullptr);

//         delete val_node_ptr_1;
//         delete val_node_ptr_2;
//         delete val_node_ptr_3;

//     }

// };

// TEST_F(ColVal_DblLink_Test, TestConstruction) {
//     // TestConstruction<__half>();
//     // TestConstruction<float>();
//     // TestConstruction<double>();
// }

// // TEST_F(ColVal_DblLink_Test, TestChaining) {
// //     TestChaining<__half>();
// //     TestChaining<float>();
// //     TestChaining<double>();
// // }

// // TEST_F(ColVal_DblLink_Test, DeleteMiddleofChain) {
// //     DeleteMiddleofChain<__half>();
// //     DeleteMiddleofChain<float>();
// //     DeleteMiddleofChain<double>();
// // }

// // TEST_F(ColVal_DblLink_Test, DeleteTopofChain) {
// //     DeleteTopofChain<__half>();
// //     DeleteTopofChain<float>();
// //     DeleteTopofChain<double>();
// // }

// // TEST_F(ColVal_DblLink_Test, DeleteBottomofChain) {
// //     DeleteBottomofChain<__half>();
// //     DeleteBottomofChain<float>();
// //     DeleteBottomofChain<double>();
// // }

// // class ColValHeadManager_Test: public TestBase
// // {
// // public:

// //     template <typename T>
// //     void ConstructShallowPopulation() {

// //         ColValHeadManager<T> manager(4);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(0, 1, static_cast<T>(0.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(0, 2, static_cast<T>(10.2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(0, 3, static_cast<T>(-0.5), &manager);

// //         manager.add_head(val_node_ptr_1);
// //         manager.add_head(val_node_ptr_2);
// //         manager.add_head(val_node_ptr_3);
// //         manager.add_head(val_node_ptr_4);

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(-1.5));
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(0.));
// //         ASSERT_EQ(manager.heads[2]->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(-0.5));

// //         delete val_node_ptr_1;
// //         delete val_node_ptr_2;
// //         delete val_node_ptr_3;
// //         delete val_node_ptr_4;

// //     }

// //     template <typename T>
// //     void ConstructDeepPopulation() {

// //         ColValHeadManager<T> manager(4);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 0, static_cast<T>(0.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 0, static_cast<T>(10.2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(0, 2, static_cast<T>(-0.5), &manager);

// //         manager.add_head(val_node_ptr_1);
// //         val_node_ptr_1->connect(val_node_ptr_2);
// //         val_node_ptr_2->connect(val_node_ptr_3);
// //         manager.add_head(val_node_ptr_4);

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(-1.5));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->val, static_cast<T>(0.));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->next_col_val->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[1], nullptr);
// //         ASSERT_EQ(manager.heads[2]->val, static_cast<T>(-0.5));
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         delete val_node_ptr_1;
// //         delete val_node_ptr_2;
// //         delete val_node_ptr_3;
// //         delete val_node_ptr_4;

// //     }

// //     template <typename T>
// //     void DeleteNotHeadInChain() {

// //         ColValHeadManager<T> manager(4);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 0, static_cast<T>(0.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 0, static_cast<T>(10.2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(0, 2, static_cast<T>(-0.5), &manager);

// //         manager.add_head(val_node_ptr_1);
// //         val_node_ptr_1->connect(val_node_ptr_2);
// //         val_node_ptr_2->connect(val_node_ptr_3);
// //         manager.add_head(val_node_ptr_4);

// //         delete val_node_ptr_2;

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(-1.5));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[1], nullptr);
// //         ASSERT_EQ(manager.heads[2]->val, static_cast<T>(-0.5));
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         delete val_node_ptr_3;

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(-1.5));
// //         ASSERT_EQ(manager.heads[0]->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[1], nullptr);
// //         ASSERT_EQ(manager.heads[2]->val, static_cast<T>(-0.5));
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         delete val_node_ptr_1;
// //         delete val_node_ptr_4;

// //     }

// //     template <typename T>
// //     void DeleteHeads() {

// //         ColValHeadManager<T> manager(4);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 1, static_cast<T>(-1.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 1, static_cast<T>(0.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 1, static_cast<T>(10.2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(0, 3, static_cast<T>(-0.5), &manager);

// //         manager.add_head(val_node_ptr_1);
// //         val_node_ptr_1->connect(val_node_ptr_2);
// //         val_node_ptr_2->connect(val_node_ptr_3);
// //         manager.add_head(val_node_ptr_4);

// //         delete val_node_ptr_1;

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(0.));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(-0.5));

// //         delete val_node_ptr_4;

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(0.));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         delete val_node_ptr_2;

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[1]->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         delete val_node_ptr_3;

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1], nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //     }

// //     template <typename T>
// //     void MixedDeletesAndAdds() {

// //         ColValHeadManager<T> manager(4);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 1, static_cast<T>(-1.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(1, 1, static_cast<T>(0.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(2, 1, static_cast<T>(10.2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(5, 1, static_cast<T>(5.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_5 = new ColVal_DblLink<T>(7, 1, static_cast<T>(8.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_6 = new ColVal_DblLink<T>(0, 3, static_cast<T>(-0.5), &manager);

// //         manager.add_head(val_node_ptr_1);
// //         val_node_ptr_1->connect(val_node_ptr_2);
// //         manager.add_head(val_node_ptr_6);

// //         delete val_node_ptr_1;
// //         delete val_node_ptr_2;

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1], nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(-0.5));

// //         manager.add_head(val_node_ptr_3);
// //         val_node_ptr_3->connect(val_node_ptr_4);

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(5.));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(-0.5));

// //         delete val_node_ptr_6;

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(10.2));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(5.));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         delete val_node_ptr_3;

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(5.));
// //         ASSERT_EQ(manager.heads[1]->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         val_node_ptr_4->connect(val_node_ptr_5);

// //         ASSERT_EQ(manager.heads[0], nullptr);
// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(5.));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(8.));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);
// //         ASSERT_EQ(manager.heads[2], nullptr);
// //         ASSERT_EQ(manager.heads[3], nullptr);

// //         delete val_node_ptr_4;
// //         delete val_node_ptr_5;

// //     }

// //     void TestBadHeadAdds() {

// //         ColValHeadManager<double> manager(4);

// //         ColVal_DblLink<double> *val_node_ptr_1 = new ColVal_DblLink<double>(0, 1, -1.5, &manager);
// //         ColVal_DblLink<double> *val_node_ptr_2 = new ColVal_DblLink<double>(1, 1, 0., &manager);
// //         ColVal_DblLink<double> *val_node_ptr_3 = new ColVal_DblLink<double>(2, 1, 10.2, &manager);
// //         ColVal_DblLink<double> *val_node_ptr_4 = new ColVal_DblLink<double>(5, 1, 5., &manager);

// //         val_node_ptr_1->connect(val_node_ptr_2);

// //         CHECK_FUNC_HAS_RUNTIME_ERROR(
// //             print_errors,
// //             [&]() {
// //                 manager.add_head(val_node_ptr_1);
// //                 manager.add_head(val_node_ptr_3);
// //             }
// //         );

// //         ColValHeadManager<double> manager2(4);

// //         CHECK_FUNC_HAS_RUNTIME_ERROR(
// //             print_errors,
// //             [&]() {
// //                 manager2.add_head(val_node_ptr_2);
// //             }
// //         );
        
// //         delete val_node_ptr_1;
// //         delete val_node_ptr_2;
// //         delete val_node_ptr_3;
// //         delete val_node_ptr_4;

// //     }

// //     void TestBadColVal() {

// //         ColValHeadManager<double> manager(2);

// //         ColVal_DblLink<double> *val_node_ptr_1 = new ColVal_DblLink<double>(0, 0, -1.5, &manager);
// //         ColVal_DblLink<double> *val_node_ptr_2 = new ColVal_DblLink<double>(1, 1, 0., &manager);
// //         ColVal_DblLink<double> *val_node_ptr_3;

// //         CHECK_FUNC_HAS_RUNTIME_ERROR(
// //             print_errors,
// //             [&]() {
// //                 val_node_ptr_3 = new ColVal_DblLink<double>(1, 2, 0., &manager);
// //                 delete val_node_ptr_3;
// //             }
// //         );

// //         manager.add_head(val_node_ptr_1);
// //         manager.add_head(val_node_ptr_2);

// //         CHECK_FUNC_HAS_RUNTIME_ERROR(
// //             print_errors,
// //             [&]() {
// //                 val_node_ptr_3 = new ColVal_DblLink<double>(1, 2, 0., &manager);
// //                 delete val_node_ptr_3;
// //             }
// //         );

// //         delete val_node_ptr_1;
// //         delete val_node_ptr_2;

// //     }

// // };

// // TEST_F(ColValHeadManager_Test, ConstructShallowPopulation) {
// //     ConstructShallowPopulation<__half>();
// //     ConstructShallowPopulation<float>();
// //     ConstructShallowPopulation<double>();
// // }

// // TEST_F(ColValHeadManager_Test, ConstructDeepPopulation) {
// //     ConstructDeepPopulation<__half>();
// //     ConstructDeepPopulation<float>();
// //     ConstructDeepPopulation<double>();
// // }

// // TEST_F(ColValHeadManager_Test, DeleteNotHeadInChain) {
// //     DeleteNotHeadInChain<__half>();
// //     DeleteNotHeadInChain<float>();
// //     DeleteNotHeadInChain<double>();
// // }

// // TEST_F(ColValHeadManager_Test, DeleteHeads) {
// //     DeleteHeads<__half>();
// //     DeleteHeads<float>();
// //     DeleteHeads<double>();
// // }

// // TEST_F(ColValHeadManager_Test, MixedDeletesAndAdds) {
// //     MixedDeletesAndAdds<__half>();
// //     MixedDeletesAndAdds<float>();
// //     MixedDeletesAndAdds<double>();
// // }

// // TEST_F(ColValHeadManager_Test, TestBadHeadAdds) {
// //     TestBadHeadAdds();
// // }

// // TEST_F(ColValHeadManager_Test, TestBadColVal) {
// //     TestBadColVal();
// // }

// // class PSizeRowHeapTest: public TestBase
// // {
// // public:

// //     template <typename T>
// //     void FilledRowConstruction() {

// //         ColValHeadManager<T> manager(5);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(0, 1, static_cast<T>(0.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(0, 2, static_cast<T>(10.2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(0, 3, static_cast<T>(5.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_5 = new ColVal_DblLink<T>(0, 4, static_cast<T>(-0.5), &manager);

// //         PSizeRowHeap<T> row_0(0, 5);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_1);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_2);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_3);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_4);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_5);

// //         ASSERT_EQ(row_0.heap.size(), 5);
// //         ASSERT_EQ(row_0.heap[0]->val, static_cast<T>(-1.5));
// //         ASSERT_LT(row_0.heap[0]->val, row_0.heap[1]->val);
// //         ASSERT_LT(row_0.heap[0]->val, row_0.heap[2]->val);
// //         ASSERT_LT(row_0.heap[1]->val, row_0.heap[3]->val);
// //         ASSERT_LT(row_0.heap[1]->val, row_0.heap[4]->val);

// //         delete val_node_ptr_1;
// //         delete val_node_ptr_2;
// //         delete val_node_ptr_3;
// //         delete val_node_ptr_4;
// //         delete val_node_ptr_5;

// //     }

// //     template <typename T>
// //     void OverflowRowBasic() {

// //         ColValHeadManager<T> manager(5);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(-1.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(0, 1, static_cast<T>(0.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(0, 2, static_cast<T>(10.2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(0, 3, static_cast<T>(5.), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_5 = new ColVal_DblLink<T>(0, 4, static_cast<T>(-0.5), &manager);

// //         PSizeRowHeap<T> row_0(0, 3);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_1);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_2);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_3);

// //         ASSERT_EQ(row_0.heap.size(), 3);
// //         ASSERT_EQ(row_0.heap[0]->val, static_cast<T>(-1.5));
// //         ASSERT_TRUE(((row_0.heap[1]->val == static_cast<T>(0.)) ||
// //                      (row_0.heap[1]->val == static_cast<T>(10.2))));
// //         ASSERT_TRUE(((row_0.heap[2]->val == static_cast<T>(0.)) ||
// //                      (row_0.heap[2]->val == static_cast<T>(10.2))));

// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_4);

// //         ASSERT_EQ(row_0.heap[0]->val, static_cast<T>(0.));
// //         ASSERT_TRUE(((row_0.heap[1]->val == static_cast<T>(5.)) ||
// //                      (row_0.heap[1]->val == static_cast<T>(10.2))));
// //         ASSERT_TRUE(((row_0.heap[2]->val == static_cast<T>(5.)) ||
// //                      (row_0.heap[2]->val == static_cast<T>(10.2))));

// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_5);

// //         ASSERT_EQ(row_0.heap[0]->val, static_cast<T>(0.));
// //         ASSERT_TRUE(((row_0.heap[1]->val == static_cast<T>(5.)) ||
// //                      (row_0.heap[1]->val == static_cast<T>(10.2))));
// //         ASSERT_TRUE(((row_0.heap[2]->val == static_cast<T>(5.)) ||
// //                      (row_0.heap[2]->val == static_cast<T>(10.2))));

// //         delete val_node_ptr_2;
// //         delete val_node_ptr_3;
// //         delete val_node_ptr_4;

// //     }

// //     template <typename T>
// //     void OverflowRowMultiplePassdown() {

// //         ColValHeadManager<T> manager(6);

// //         ColVal_DblLink<T> *val_node_ptr_1 = new ColVal_DblLink<T>(0, 0, static_cast<T>(1), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2 = new ColVal_DblLink<T>(0, 1, static_cast<T>(2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3 = new ColVal_DblLink<T>(0, 2, static_cast<T>(3), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4 = new ColVal_DblLink<T>(0, 3, static_cast<T>(4), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_5 = new ColVal_DblLink<T>(0, 4, static_cast<T>(5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_6 = new ColVal_DblLink<T>(0, 5, static_cast<T>(6), &manager);

// //         PSizeRowHeap<T> row_0(0, 4);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_1);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_2);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_3);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_4);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_5);
// //         row_0.attempt_add_and_delete_min_val(val_node_ptr_6);

// //         ASSERT_EQ(row_0.heap.size(), 4);
// //         ASSERT_EQ(row_0.heap[0]->val, static_cast<T>(3));
// //         ASSERT_LT(row_0.heap[0]->val, row_0.heap[1]->val);
// //         ASSERT_LT(row_0.heap[0]->val, row_0.heap[2]->val);
// //         ASSERT_LT(row_0.heap[1]->val, row_0.heap[3]->val);

// //         delete val_node_ptr_3;
// //         delete val_node_ptr_4;
// //         delete val_node_ptr_5;
// //         delete val_node_ptr_6;

// //     }

// // };

// // TEST_F(PSizeRowHeapTest, FilledRowConstruction) {
// //     FilledRowConstruction<__half>();
// //     FilledRowConstruction<float>();
// //     FilledRowConstruction<double>();
// // }

// // TEST_F(PSizeRowHeapTest, OverflowRowBasic) {
// //     OverflowRowBasic<__half>();
// //     OverflowRowBasic<float>();
// //     OverflowRowBasic<double>();
// // }

// // TEST_F(PSizeRowHeapTest, OverflowRowMultiplePassdown) {
// //     OverflowRowMultiplePassdown<__half>();
// //     OverflowRowMultiplePassdown<float>();
// //     OverflowRowMultiplePassdown<double>();
// // }

// // class PSizeRowHeap_ColumnHeadManager_Interaction_Test: public TestBase
// // {
// // public:
    
// //     template <typename T>
// //     void TestCommonFillNoDeletions() {

// //         ColValHeadManager<T> manager(5);

// //         std::vector<PSizeRowHeap<T>> rows;
// //         for (int i=0; i<7; ++i) { rows.push_back(PSizeRowHeap<T>(i, 5)); }

// //         ColVal_DblLink<T> *val_node_ptr_0_0 = new ColVal_DblLink<T>(0, 0, static_cast<T>(1), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3_0 = new ColVal_DblLink<T>(3, 0, static_cast<T>(2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4_0 = new ColVal_DblLink<T>(4, 0, static_cast<T>(3), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_6_0 = new ColVal_DblLink<T>(6, 0, static_cast<T>(4), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2_1 = new ColVal_DblLink<T>(2, 1, static_cast<T>(5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_3_3 = new ColVal_DblLink<T>(3, 3, static_cast<T>(6), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_4_3 = new ColVal_DblLink<T>(4, 3, static_cast<T>(7), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_5_3 = new ColVal_DblLink<T>(5, 3, static_cast<T>(8), &manager);
        
// //         val_node_ptr_0_0->connect(val_node_ptr_3_0);
// //         val_node_ptr_3_0->connect(val_node_ptr_4_0);
// //         val_node_ptr_4_0->connect(val_node_ptr_6_0);
// //         manager.add_head(val_node_ptr_0_0);

// //         manager.add_head(val_node_ptr_2_1);

// //         val_node_ptr_3_3->connect(val_node_ptr_4_3);
// //         val_node_ptr_4_3->connect(val_node_ptr_5_3);
// //         manager.add_head(val_node_ptr_3_3);

// //         rows[0].attempt_add_and_delete_min_val(val_node_ptr_0_0);
// //         rows[2].attempt_add_and_delete_min_val(val_node_ptr_2_1);
// //         rows[3].attempt_add_and_delete_min_val(val_node_ptr_3_0);
// //         rows[3].attempt_add_and_delete_min_val(val_node_ptr_3_3);
// //         rows[4].attempt_add_and_delete_min_val(val_node_ptr_4_0);
// //         rows[4].attempt_add_and_delete_min_val(val_node_ptr_4_3);
// //         rows[5].attempt_add_and_delete_min_val(val_node_ptr_5_3);
// //         rows[6].attempt_add_and_delete_min_val(val_node_ptr_6_0);

// //         delete val_node_ptr_0_0;
// //         delete val_node_ptr_3_0;
// //         delete val_node_ptr_4_0;
// //         delete val_node_ptr_6_0;
// //         delete val_node_ptr_2_1;
// //         delete val_node_ptr_3_3;
// //         delete val_node_ptr_4_3;
// //         delete val_node_ptr_5_3;

// //     }

// //     template <typename T>
// //     void TestCommonFillWithDeletions() {

// //         const int m(3);
// //         const int n(7);
// //         const int p(4);

// //         ColValHeadManager<T> manager(7);

// //         std::vector<PSizeRowHeap<T>> rows;
// //         for (int i=0; i<m; ++i) { rows.push_back(PSizeRowHeap<T>(i, p)); }

// //         ColVal_DblLink<T> *val_node_ptr_0_0 = new ColVal_DblLink<T>(0, 0, static_cast<T>(1), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_1_0 = new ColVal_DblLink<T>(1, 0, static_cast<T>(2), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2_0 = new ColVal_DblLink<T>(2, 0, static_cast<T>(3), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_1_1 = new ColVal_DblLink<T>(1, 1, static_cast<T>(4), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_2_1 = new ColVal_DblLink<T>(2, 1, static_cast<T>(5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_1_2 = new ColVal_DblLink<T>(1, 2, static_cast<T>(2.5), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_1_3 = new ColVal_DblLink<T>(1, 3, static_cast<T>(7), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_1_4 = new ColVal_DblLink<T>(1, 4, static_cast<T>(8), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_1_5 = new ColVal_DblLink<T>(1, 5, static_cast<T>(9), &manager);
// //         ColVal_DblLink<T> *val_node_ptr_1_6 = new ColVal_DblLink<T>(1, 6, static_cast<T>(10), &manager);

// //         val_node_ptr_0_0->connect(val_node_ptr_1_0);
// //         val_node_ptr_1_0->connect(val_node_ptr_2_0);
// //         manager.add_head(val_node_ptr_0_0);

// //         val_node_ptr_1_1->connect(val_node_ptr_2_1);
// //         manager.add_head(val_node_ptr_1_1);

// //         manager.add_head(val_node_ptr_1_2);
// //         manager.add_head(val_node_ptr_1_3);

// //         rows[0].attempt_add_and_delete_min_val(val_node_ptr_0_0);
// //         rows[1].attempt_add_and_delete_min_val(val_node_ptr_1_0);
// //         rows[2].attempt_add_and_delete_min_val(val_node_ptr_2_0);

// //         rows[1].attempt_add_and_delete_min_val(val_node_ptr_1_1);
// //         rows[2].attempt_add_and_delete_min_val(val_node_ptr_2_1);

// //         rows[1].attempt_add_and_delete_min_val(val_node_ptr_1_2);
// //         rows[1].attempt_add_and_delete_min_val(val_node_ptr_1_3);

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(1));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->val, static_cast<T>(2));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->next_col_val->val, static_cast<T>(3));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->next_col_val->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(4));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(5));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[2]->val, static_cast<T>(2.5));
// //         ASSERT_EQ(manager.heads[2]->next_col_val, nullptr);
        
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(7));
// //         ASSERT_EQ(manager.heads[3]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[4], nullptr);
// //         ASSERT_EQ(manager.heads[5], nullptr);
// //         ASSERT_EQ(manager.heads[6], nullptr);

// //         manager.add_head(val_node_ptr_1_4);
// //         rows[1].attempt_add_and_delete_min_val(val_node_ptr_1_4);

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(1));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->val, static_cast<T>(3));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(4));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(5));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[2]->val, static_cast<T>(2.5));
// //         ASSERT_EQ(manager.heads[2]->next_col_val, nullptr);
        
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(7));
// //         ASSERT_EQ(manager.heads[3]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[4]->val, static_cast<T>(8.));
// //         ASSERT_EQ(manager.heads[4]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[5], nullptr);
// //         ASSERT_EQ(manager.heads[6], nullptr);

// //         manager.add_head(val_node_ptr_1_5);
// //         rows[1].attempt_add_and_delete_min_val(val_node_ptr_1_5);

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(1));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->val, static_cast<T>(3));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(4));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->val, static_cast<T>(5));
// //         ASSERT_EQ(manager.heads[1]->next_col_val->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[2], nullptr);
        
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(7));
// //         ASSERT_EQ(manager.heads[3]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[4]->val, static_cast<T>(8.));
// //         ASSERT_EQ(manager.heads[4]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[5]->val, static_cast<T>(9.));
// //         ASSERT_EQ(manager.heads[5]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[6], nullptr);

// //         manager.add_head(val_node_ptr_1_6);
// //         rows[1].attempt_add_and_delete_min_val(val_node_ptr_1_6);

// //         ASSERT_EQ(manager.heads[0]->val, static_cast<T>(1));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->val, static_cast<T>(3));
// //         ASSERT_EQ(manager.heads[0]->next_col_val->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[1]->val, static_cast<T>(5));
// //         ASSERT_EQ(manager.heads[1]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[2], nullptr);
        
// //         ASSERT_EQ(manager.heads[3]->val, static_cast<T>(7));
// //         ASSERT_EQ(manager.heads[3]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[4]->val, static_cast<T>(8.));
// //         ASSERT_EQ(manager.heads[4]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[5]->val, static_cast<T>(9.));
// //         ASSERT_EQ(manager.heads[5]->next_col_val, nullptr);

// //         ASSERT_EQ(manager.heads[6]->val, static_cast<T>(10.));
// //         ASSERT_EQ(manager.heads[6]->next_col_val, nullptr);

// //         delete val_node_ptr_0_0;
// //         delete val_node_ptr_2_0;
// //         delete val_node_ptr_2_1;
// //         delete val_node_ptr_1_3;
// //         delete val_node_ptr_1_4;
// //         delete val_node_ptr_1_5;
// //         delete val_node_ptr_1_6;

// //     }

// //     template <typename T>
// //     void TestCommonFillWithAdjRowDeletions() {

// //     }

// // };

// // TEST_F(PSizeRowHeap_ColumnHeadManager_Interaction_Test, TestCommonFillNoDeletions) {
// //     TestCommonFillNoDeletions<__half>();
// //     TestCommonFillNoDeletions<float>();
// //     TestCommonFillNoDeletions<double>();
// // }

// // TEST_F(PSizeRowHeap_ColumnHeadManager_Interaction_Test, TestCommonFillWithDeletions) {
// //     TestCommonFillWithDeletions<__half>();
// //     TestCommonFillWithDeletions<float>();
// //     TestCommonFillWithDeletions<double>();
// // }

// // TEST_F(PSizeRowHeap_ColumnHeadManager_Interaction_Test, TestCommonFillWithAdjRowDeletions) {
// //     TestCommonFillWithAdjRowDeletions<__half>();
// //     TestCommonFillWithAdjRowDeletions<float>();
// //     TestCommonFillWithAdjRowDeletions<double>();
// // }