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

    template <typename T>
    void TestCoeffAccess() { TestCoeffAccess_Base<MatrixDense, T>(); }

    template <typename T>
    void TestStaticCreation() { TestStaticCreation_Base<MatrixDense, T>(); }

    template <typename T>
    void TestCol() { TestCol_Base<MatrixDense, T>(); }

    template <typename T>
    void TestBlock() {

        TestBlock_Base<MatrixDense, T>();

        const MatrixDense<T> const_mat ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
            {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(10)},
            {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14), static_cast<T>(15)},
            {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(20)}
        });
        MatrixDense<T> mat(const_mat);

        // Test assignment from MatrixDense
        MatrixDense<T> zero_2_3(MatrixDense<T>::Zero(2, 3));
        mat.block(1, 1, 2, 3) = zero_2_3;
        MatrixDense<T> test_assign_2_3 ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
            {static_cast<T>(6), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(10)},
            {static_cast<T>(11), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(15)},
            {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(20)}
        });
        ASSERT_MATRIX_EQ(mat, test_assign_2_3);

        // Test assignment from MatrixVector

        // // Test assignment
        // MatrixVector<T> assign_vec({static_cast<T>(1),
        //                             static_cast<T>(1),
        //                             static_cast<T>(1),
        //                             static_cast<T>(1)});
        // mat.col(2) = assign_vec;
        // for (int j=0; j<2; ++j) {
        //     for (int i=0; i<4; ++i) {
        //         ASSERT_EQ(mat.coeff(i, j), const_mat.coeff(i, j));
        //     }
        // }
        // for (int i=0; i<4; ++i) { ASSERT_EQ(mat.coeff(i, 2), static_cast<T>(1)); }

    }

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

TEST_F(MatrixDense_Test, TestCoeffAccess) {
    TestCoeffAccess<half>();
    TestCoeffAccess<float>();
    TestCoeffAccess<double>();
}

TEST_F(MatrixDense_Test, TestStaticCreation) {
    TestStaticCreation<half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(MatrixDense_Test, TestCol) {
    TestCol<half>();
    TestCol<float>();
    TestCol<double>();
}

TEST_F(MatrixDense_Test, TestBlock) {
    TestBlock<half>();
    TestBlock<float>();
    TestBlock<double>();
}
