#include "test.h"

#include "tools/DenseConverter.h"
#include "tools/read_matrix.h"

// General matrix read tests
class read_matrixCSV_General_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void ReadEmptyMatrix() {

        fs::path empty_file(read_matrix_dir / fs::path("empty.csv"));
        M<double> test_empty(read_matrixCSV<M, double>(TestBase::bundle, empty_file));
        ASSERT_EQ(test_empty.rows(), 0);
        ASSERT_EQ(test_empty.cols(), 0);

    }

    template <template <typename> typename M>
    void ReadBadMatrices() {

        // Try to load non-existent file
        fs::path bad_file_0(read_matrix_dir / fs::path("thisfile"));
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { M<double> test(read_matrixCSV<M, double>(TestBase::bundle, bad_file_0)); }
        );

        // Try to load file with too small row
        fs::path bad_file_1(read_matrix_dir / fs::path("bad1.csv"));
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { M<double> test(read_matrixCSV<M, double>(TestBase::bundle, bad_file_1)); }
        );

        // Try to load file with too big row
        fs::path bad_file_2(read_matrix_dir / fs::path("bad2.csv"));
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { M<double> test(read_matrixCSV<M, double>(TestBase::bundle, bad_file_2)); }
        );

        // Try to load file with invalid character argument
        fs::path bad_file_3(read_matrix_dir / fs::path("bad3.csv"));
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { M<double> test(read_matrixCSV<M, double>(TestBase::bundle, bad_file_3)); }
        );


    }

};

TEST_F(read_matrixCSV_General_Test, ReadEmptyMatrix) {
    ReadEmptyMatrix<MatrixDense>();
    ReadEmptyMatrix<NoFillMatrixSparse>();
}

TEST_F(read_matrixCSV_General_Test, ReadBadFiles) {
    ReadBadMatrices<MatrixDense>();
    ReadBadMatrices<NoFillMatrixSparse>();
}

template <typename T>
class read_matrixCSV_T_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void ReadSquareMatrix() {

        M<T> target1(
            TestBase::bundle,
            {{static_cast<T>(1),  static_cast<T>(2),  static_cast<T>(3)},
             {static_cast<T>(4),  static_cast<T>(5),  static_cast<T>(6)},
             {static_cast<T>(7),  static_cast<T>(8),  static_cast<T>(9)}}
        );

        M<T> target2(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(10)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(20)},
             {static_cast<T>(21), static_cast<T>(22), static_cast<T>(23), static_cast<T>(24), static_cast<T>(25)}}
        );
    
        fs::path square1_file(read_matrix_dir / fs::path("square1.csv"));
        fs::path square2_file(read_matrix_dir / fs::path("square2.csv"));
        M<T> test1(read_matrixCSV<M, T>(TestBase::bundle, square1_file));
        M<T> test2(read_matrixCSV<M, T>(TestBase::bundle, square2_file));

        ASSERT_MATRIX_NEAR(test1, target1, static_cast<T>(Tol<T>::roundoff()));
        ASSERT_MATRIX_NEAR(test2, target2, static_cast<T>(Tol<T>::roundoff()));

    }

    template <template <typename> typename M>
    void ReadWideTallMatrix() {

        M<T> target_wide(
            TestBase::bundle,
            {{static_cast<T>(10), static_cast<T>(9), static_cast<T>(8), static_cast<T>(7), static_cast<T>(6)},
             {static_cast<T>(5), static_cast<T>(4), static_cast<T>(3), static_cast<T>(2), static_cast<T>(1)}}
        );

        M<T> target_tall(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2)},
             {static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8)}}
        );

        fs::path wide_file(read_matrix_dir / fs::path("wide.csv"));
        fs::path tall_file(read_matrix_dir / fs::path("tall.csv"));

        M<T> test_wide(read_matrixCSV<M, T>(TestBase::bundle, wide_file));
        M<T> test_tall(read_matrixCSV<M, T>(TestBase::bundle, tall_file));

        ASSERT_MATRIX_NEAR(test_wide, target_wide, static_cast<T>(Tol<T>::roundoff()));
        ASSERT_MATRIX_NEAR(test_tall, target_tall, static_cast<T>(Tol<T>::roundoff()));

    }

    template <template <typename> typename M>
    void ReadPrecise(
        fs::path precise_file,
        MatrixDense<T> target_precise
    ) {

        M<T> test_precise(read_matrixCSV<M, T>(TestBase::bundle, precise_file));
        DenseConverter<M, T> converter;
        ASSERT_MATRIX_NEAR(
            test_precise,
            converter.convert_matrix(target_precise),
            static_cast<T>(Tol<T>::roundoff())
        );

    }

    template <template <typename> typename M>
    void ReadDifferentThanPrecise(
        fs::path precise_file,
        MatrixDense<T> target_precise
    ) {

        T eps(static_cast<T>(1.5)*static_cast<T>(Tol<T>::roundoff()));
        DenseConverter<M, T> converter;
        M<T> miss_precise_up(
            converter.convert_matrix(target_precise + MatrixDense<T>::Ones(TestBase::bundle, 2, 2)*eps)
        );
        M<T> miss_precise_down(
            converter.convert_matrix(target_precise - MatrixDense<T>::Ones(TestBase::bundle, 2, 2)*eps)
        );

        M<T> test_precise(read_matrixCSV<M, T>(TestBase::bundle, precise_file));
        ASSERT_MATRIX_LT(test_precise, miss_precise_up);
        ASSERT_MATRIX_GT(test_precise, miss_precise_down);

    }

};

// All type vector read tests
class read_matrixCSV_Vector_Test: public TestBase
{
public:

    template <typename T>
    void ReadVector() {

        Vector<T> target(TestBase::bundle, {1, 2, 3, 4, 5, 6});

        fs::path vector_file(read_matrix_dir / fs::path("vector.csv"));
        Vector<T> test(read_matrixCSV<Vector, T>(TestBase::bundle, vector_file));

        ASSERT_VECTOR_NEAR(test, target, static_cast<T>(Tol<T>::roundoff()));

    }

};

TEST_F(read_matrixCSV_Vector_Test, ReadDoubleVector) { ReadVector<double>(); }
TEST_F(read_matrixCSV_Vector_Test, ReadSingleVector) { ReadVector<float>(); }
TEST_F(read_matrixCSV_Vector_Test, ReadHalfVector) { ReadVector<__half>(); }

TEST_F(read_matrixCSV_Vector_Test, FailOnMatrix) {    
    fs::path mat(read_matrix_dir / fs::path("square1.csv"));
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [=]() { Vector<double> test(read_matrixCSV<Vector, double>(TestBase::bundle, mat)); }
    );
}

// Double type matrix read tests
class read_matrixCSV_Double_Test: public read_matrixCSV_T_Test<double> {};

TEST_F(read_matrixCSV_Double_Test, ReadSquareMatrix) {
    ReadSquareMatrix<MatrixDense>();
    ReadSquareMatrix<NoFillMatrixSparse>();
}

TEST_F(read_matrixCSV_Double_Test, ReadWideTallMatrix) {
    ReadWideTallMatrix<MatrixDense>();
    ReadWideTallMatrix<NoFillMatrixSparse>();
}

TEST_F(read_matrixCSV_Double_Test, ReadPreciseMatrix) {

    fs::path precise_file(read_matrix_dir / fs::path("double_precise.csv"));

    MatrixDense<double> target(
        TestBase::bundle,
        {{1.12345678901232, 1.12345678901234},
         {1.12345678901236, 1.12345678901238}}
    );

    ReadPrecise<MatrixDense>(precise_file, target);
    ReadPrecise<NoFillMatrixSparse>(precise_file, target);

}

TEST_F(read_matrixCSV_Double_Test, ReadDifferentThanPreciseMatrix) {

    fs::path precise_file(read_matrix_dir / fs::path("double_precise.csv"));

    MatrixDense<double> target(
        TestBase::bundle,
        {{1.12345678901232, 1.12345678901234},
         {1.12345678901236, 1.12345678901238}}
    );

    ReadDifferentThanPrecise<MatrixDense>(precise_file, target);
    ReadDifferentThanPrecise<NoFillMatrixSparse>(precise_file, target);

}

TEST_F(read_matrixCSV_Double_Test, ReadPreciseMatrixDoubleLimit) {

    fs::path precise_file(read_matrix_dir / fs::path("double_precise_manual.csv"));

    MatrixDense<double> target(
        TestBase::bundle,
        {{1.1234567890123452, 1.1234567890123454},
         {1.1234567890123456, 1.1234567890123458}}
    );
    
    ReadPrecise<MatrixDense>(precise_file, target);
    ReadPrecise<NoFillMatrixSparse>(precise_file, target);

}

TEST_F(read_matrixCSV_Double_Test, ReadDifferentThanPreciseMatrixDoubleLimit) {

    fs::path precise_file(read_matrix_dir / fs::path("double_precise_manual.csv"));

    MatrixDense<double> target(
        TestBase::bundle,
        {{1.1234567890123452, 1.1234567890123454},
         {1.1234567890123456, 1.1234567890123458}}
    );

    ReadDifferentThanPrecise<MatrixDense>(precise_file, target);
    ReadDifferentThanPrecise<NoFillMatrixSparse>(precise_file, target);

}

// Single type matrix read tests
class read_matrixCSV_Single_Test: public read_matrixCSV_T_Test<float> {};

TEST_F(read_matrixCSV_Single_Test, ReadSquareMatrix) {
    ReadSquareMatrix<MatrixDense>();
    ReadSquareMatrix<NoFillMatrixSparse>();
}

TEST_F(read_matrixCSV_Single_Test, ReadWideTallMatrix) {
    ReadWideTallMatrix<MatrixDense>();
    ReadWideTallMatrix<NoFillMatrixSparse>();
}

TEST_F(read_matrixCSV_Single_Test, ReadPreciseMatrix) {

    fs::path precise_file(read_matrix_dir / fs::path("single_precise.csv"));

    MatrixDense<float> target(
        TestBase::bundle,
        {{static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
         {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}}
    );

    ReadPrecise<MatrixDense>(precise_file, target);
    ReadPrecise<NoFillMatrixSparse>(precise_file, target);

}

TEST_F(read_matrixCSV_Single_Test, ReadDifferentThanPreciseMatrix) {

    fs::path precise_file(read_matrix_dir / fs::path("single_precise.csv"));

    MatrixDense<float> target(
        TestBase::bundle,
        {{static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
         {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}}
    );

    ReadDifferentThanPrecise<MatrixDense>(precise_file, target);
    ReadDifferentThanPrecise<NoFillMatrixSparse>(precise_file, target);

}

// Half type matrix read tests
class read_matrixCSV_Half_Test: public read_matrixCSV_T_Test<__half> {};

TEST_F(read_matrixCSV_Half_Test, ReadSquareMatrix) {
    ReadSquareMatrix<MatrixDense>();
    ReadSquareMatrix<NoFillMatrixSparse>();
}

TEST_F(read_matrixCSV_Half_Test, ReadWideTallMatrix) {
    ReadWideTallMatrix<MatrixDense>();
    ReadWideTallMatrix<NoFillMatrixSparse>();
}

TEST_F(read_matrixCSV_Half_Test, ReadPreciseMatrix) {

    fs::path precise_file(read_matrix_dir / fs::path("half_precise.csv"));

    MatrixDense<__half> target(
        TestBase::bundle,
        {{static_cast<__half>(1.123), static_cast<__half>(1.124)},
         {static_cast<__half>(1.125), static_cast<__half>(1.126)}}
    );

    ReadPrecise<MatrixDense>(precise_file, target);
    ReadPrecise<NoFillMatrixSparse>(precise_file, target);

}

TEST_F(read_matrixCSV_Half_Test, ReadDifferentThanPreciseMatrix) {

    fs::path precise_file(read_matrix_dir / fs::path("half_precise.csv"));

    MatrixDense<__half> target(
        TestBase::bundle,
        {{static_cast<__half>(1.123), static_cast<__half>(1.124)},
         {static_cast<__half>(1.125), static_cast<__half>(1.126)}}
    );

    ReadDifferentThanPrecise<MatrixDense>(precise_file, target);
    ReadDifferentThanPrecise<NoFillMatrixSparse>(precise_file, target);

}