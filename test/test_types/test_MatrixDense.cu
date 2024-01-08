#include "test_Matrix.h"

#include "types/MatrixDense.h"

class MatrixDense_Test: public Matrix_Test
{
public:

    template <typename T>
    void TestConstruction() { TestConstruction_Base<MatrixDense, T>(); }

    template <typename T>
    void TestListInitialization() { TestListInitialization_Base<MatrixDense, T>(); }

    void TestBadListInitialization() { TestBadListInitialization_Base<MatrixDense>(); }

    // template <typename T>
    // void TestCoeffAccess() { TestCoeffAccess_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestStaticCreation() { TestStaticCreation_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestCol() { TestCol_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestBlock() {

    //     const MatrixDense<T> const_mat ({
    //         {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
    //         {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(10)},
    //         {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14), static_cast<T>(15)},
    //         {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(20)}
    //     });
    //     MatrixDense<T> mat(const_mat);
        
    //     // Test cast/access for block 0, 0, 3, 4
    //     MatrixDense<T> mat_0_0_3_4(mat.block(0, 0, 3, 4));
    //     MatrixDense<T> test_0_0_3_4 ({
    //         {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
    //         {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
    //         {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14)}
    //     });
    //     ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

    //     // Test cast/access for block 1, 2, 3, 1
    //     MatrixDense<T> mat_1_2_3_1(mat.block(1, 2, 3, 1));
    //     MatrixDense<T> test_1_2_3_1 ({
    //         {static_cast<T>(8)},
    //         {static_cast<T>(13)},
    //         {static_cast<T>(18)}
    //     });
    //     ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

    //     // Test assignment from MatrixDense
    //     mat = const_mat;
    //     MatrixDense<T> zero_2_3(MatrixDense<T>::Zero(2, 3));
    //     mat.block(1, 1, 2, 3) = zero_2_3;
    //     MatrixDense<T> test_assign_2_3 ({
    //         {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
    //         {static_cast<T>(6), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(10)},
    //         {static_cast<T>(11), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(15)},
    //         {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(20)}
    //     });
    //     ASSERT_MATRIX_EQ(mat, test_assign_2_3);

    //     // Test assignment from MatrixVector
    //     mat = const_mat;
    //     MatrixVector<T> assign_vec({static_cast<T>(1),
    //                                 static_cast<T>(1),
    //                                 static_cast<T>(1)});
    //     mat.block(1, 4, 3, 1) = assign_vec;
    //     MatrixDense<T> test_assign_1_4 ({
    //         {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
    //         {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(1)},
    //         {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14), static_cast<T>(1)},
    //         {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(1)}
    //     });
    //     ASSERT_MATRIX_EQ(mat, test_assign_1_4);

    // }

    // template <typename T>
    // void TestTranspose() { TestTranspose_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestScale() { TestScale_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestMatVec() { TestMatVec_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestMatMat() { TestMatMat_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestNorm() { TestNorm_Base<MatrixDense, T>(); }

    // template <typename T>
    // void TestAddSub() { TestAddSub_Base<MatrixDense, T>(); }

    // void TestCast() { TestCast_Base<MatrixDense>(); }

};

TEST_F(MatrixDense_Test, TestConstruction) {
    TestConstruction<half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(MatrixDense_Test, TestListInitialization) {
    TestListInitialization<half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixDense_Test, TestBadListInitialization) { TestBadListInitialization(); }

// TEST_F(MatrixDense_Test, TestCoeffAccess) {
//     TestCoeffAccess<half>();
//     TestCoeffAccess<float>();
//     TestCoeffAccess<double>();
// }

// TEST_F(MatrixDense_Test, TestStaticCreation) {
//     TestStaticCreation<half>();
//     TestStaticCreation<float>();
//     TestStaticCreation<double>();
// }

// TEST_F(MatrixDense_Test, TestCol) {
//     TestCol<half>();
//     TestCol<float>();
//     TestCol<double>();
// }

// TEST_F(MatrixDense_Test, TestBlock) {
//     TestBlock<half>();
//     TestBlock<float>();
//     TestBlock<double>();
// }

// TEST_F(MatrixDense_Test, TestTranspose) {
//     TestTranspose<half>();
//     TestTranspose<float>();
//     TestTranspose<double>();
// }

// TEST_F(MatrixDense_Test, TestScale) {
//     TestScale<half>();
//     TestScale<float>();
//     TestScale<double>();
// }

// TEST_F(MatrixDense_Test, TestMatVec) {
//     TestMatVec<half>();
//     TestMatVec<float>();
//     TestMatVec<double>();
// }

// TEST_F(MatrixDense_Test, TestMatMat) {
//     TestMatMat<half>();
//     TestMatMat<float>();
//     TestMatMat<double>();
// }

// TEST_F(MatrixDense_Test, TestNorm) {
//     TestNorm<half>();
//     TestNorm<float>();
//     TestNorm<double>();
// }

// TEST_F(MatrixDense_Test, TestAddSub) {
//     TestAddSub<half>();
//     TestAddSub<float>();
//     TestAddSub<double>();
// }

// TEST_F(MatrixDense_Test, TestCast) { TestCast(); }