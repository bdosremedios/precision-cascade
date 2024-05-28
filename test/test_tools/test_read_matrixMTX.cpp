#include "../test.h"

#include "tools/read_matrix.h"

// General matrix read tests
class read_matrixMTX_Test: public TestBase
{
public:

    template <typename T>
    void ReadEmptyMatrix() {

        fs::path empty_file(read_matrix_dir / fs::path("empty.mtx"));
        NoFillMatrixSparse<double> test_empty(read_matrixMTX<double>(TestBase::bundle, empty_file));
        ASSERT_EQ(test_empty.rows(), 0);
        ASSERT_EQ(test_empty.cols(), 0);

    }

    template <typename T>
    void ReadSimpleGeneral() {

        fs::path file_square(read_matrix_dir / fs::path("square.mtx"));
        NoFillMatrixSparse<T> target_square(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)}}
        );
        NoFillMatrixSparse<T> test_square(read_matrixMTX<T>(TestBase::bundle, file_square));
        ASSERT_MATRIX_EQ(test_square, target_square);

        fs::path file_tall(read_matrix_dir / fs::path("tall.mtx"));
        NoFillMatrixSparse<T> target_tall(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2)},
             {static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(5)},
             {static_cast<T>(7), static_cast<T>(8)}}
        );
        NoFillMatrixSparse<T> test_tall(read_matrixMTX<T>(TestBase::bundle, file_tall));
        ASSERT_MATRIX_EQ(test_tall, target_tall);

        fs::path file_wide(read_matrix_dir / fs::path("wide.mtx"));
        NoFillMatrixSparse<T> target_wide(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(10)}}
        );
        NoFillMatrixSparse<T> test_wide(read_matrixMTX<T>(TestBase::bundle, file_wide));
        ASSERT_MATRIX_EQ(test_wide, target_wide);

    }

    template <typename T>
    void ReadSimpleSymmetric() {

        fs::path file_symmetric(read_matrix_dir / fs::path("square_symmetric.mtx"));
        NoFillMatrixSparse<T> target_symmetric(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(2), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(3), static_cast<T>(7), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)},
             {static_cast<T>(4), static_cast<T>(8), static_cast<T>(11), static_cast<T>(13), static_cast<T>(14)},
             {static_cast<T>(5), static_cast<T>(9), static_cast<T>(12), static_cast<T>(14), static_cast<T>(15)}}
        );
        NoFillMatrixSparse<T> test_symmetric(read_matrixMTX<T>(TestBase::bundle, file_symmetric));
        ASSERT_MATRIX_EQ(test_symmetric, target_symmetric);

        fs::path file_gen_match(read_matrix_dir / fs::path("square_general_match.mtx"));
        ASSERT_MATRIX_EQ(test_symmetric, read_matrixMTX<T>(TestBase::bundle, file_gen_match));

    }

    template <typename T>
    void ReadPreciseGeneral(fs::path mtx_path, NoFillMatrixSparse<T> target) {

        NoFillMatrixSparse<T> test(read_matrixMTX<T>(TestBase::bundle, mtx_path));
        ASSERT_MATRIX_EQ(test, target);
        ASSERT_MATRIX_LT(
            test,
            NoFillMatrixSparse<T>(
                MatrixDense<T>(target) +
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*Tol<T>::roundoff()
            )
        );
        ASSERT_MATRIX_GT(
            test,
            NoFillMatrixSparse<T>(
                MatrixDense<T>(target) -
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*Tol<T>::roundoff()
            )
        );
        
    }
    
    template <typename T>
    void ReadPreciseSymmetric(fs::path sym_mtx_path, NoFillMatrixSparse<T> target) {

        NoFillMatrixSparse<T> test(read_matrixMTX<T>(TestBase::bundle, sym_mtx_path));
        ASSERT_MATRIX_EQ(test, target);
        ASSERT_MATRIX_LT(
            test,
            NoFillMatrixSparse<T>(
                MatrixDense<T>(target) +
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*Tol<T>::roundoff()
            )
        );
        ASSERT_MATRIX_GT(
            test,
            NoFillMatrixSparse<T>(
                MatrixDense<T>(target) -
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*Tol<T>::roundoff()
            )
        );
        
    }

    void BadReadMTX() {

    }

};

TEST_F(read_matrixMTX_Test, ReadEmptyMatrix) {
    ReadEmptyMatrix<__half>();
    ReadEmptyMatrix<float>();
    ReadEmptyMatrix<double>();
}

TEST_F(read_matrixMTX_Test, ReadSimpleGeneral) {
    ReadSimpleGeneral<__half>();
    ReadSimpleGeneral<float>();
    ReadSimpleGeneral<double>();
}

TEST_F(read_matrixMTX_Test, ReadSimpleSymmetric) {
    ReadSimpleSymmetric<__half>();
    ReadSimpleSymmetric<float>();
    ReadSimpleSymmetric<double>();
}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Double) {
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("double_precise.mtx"),
        NoFillMatrixSparse<double>(
            TestBase::bundle,
            {{static_cast<double>(1.1234567890123450), static_cast<double>(1.1234567890123454),
              static_cast<double>(1.1234567890123457), static_cast<double>(1.1234567890123451)},
             {static_cast<double>(1.1234567890123451), static_cast<double>(1.1234567890123455),
              static_cast<double>(1.1234567890123458), static_cast<double>(1.1234567890123452)},
             {static_cast<double>(1.1234567890123452), static_cast<double>(1.1234567890123456),
              static_cast<double>(1.1234567890123459), static_cast<double>(1.1234567890123453)},
             {static_cast<double>(1.1234567890123453), static_cast<double>(1.1234567890123457),
              static_cast<double>(1.1234567890123450), static_cast<double>(1.1234567890123453)}}
        )
    );
}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Double) {
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("double_precise_symmetric.mtx"),
        NoFillMatrixSparse<double>(
            TestBase::bundle,
            {{static_cast<double>(1.1234567890123450), static_cast<double>(1.1234567890123452),
              static_cast<double>(1.1234567890123453), static_cast<double>(1.1234567890123454)},
             {static_cast<double>(1.1234567890123451), static_cast<double>(1.1234567890123454),
              static_cast<double>(1.1234567890123455), static_cast<double>(1.1234567890123456)},
             {static_cast<double>(1.1234567890123452), static_cast<double>(1.1234567890123455),
              static_cast<double>(1.1234567890123457), static_cast<double>(1.1234567890123458)},
             {static_cast<double>(1.1234567890123453), static_cast<double>(1.1234567890123456),
              static_cast<double>(1.1234567890123458), static_cast<double>(1.1234567890123459)}}
        )
    );
}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Single) {
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("single_precise.mtx"),
        NoFillMatrixSparse<float>(
            TestBase::bundle,
            {{static_cast<float>(1.12345670), static_cast<float>(1.12345674),
              static_cast<float>(1.12345678), static_cast<float>(1.12345672)},
             {static_cast<float>(1.12345671), static_cast<float>(1.12345675),
              static_cast<float>(1.12345679), static_cast<float>(1.12345673)},
             {static_cast<float>(1.12345672), static_cast<float>(1.12345676),
              static_cast<float>(1.12345670), static_cast<float>(1.12345674)},
             {static_cast<float>(1.12345673), static_cast<float>(1.12345677),
              static_cast<float>(1.12345671), static_cast<float>(1.12345675)}}
        )
    );
}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Single) {
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("single_precise_symmetric.mtx"),
        NoFillMatrixSparse<float>(
            TestBase::bundle,
            {{static_cast<float>(1.12345670), static_cast<float>(1.12345671),
              static_cast<float>(1.12345672), static_cast<float>(1.12345673)},
             {static_cast<float>(1.12345671), static_cast<float>(1.12345674),
              static_cast<float>(1.12345675), static_cast<float>(1.12345676)},
             {static_cast<float>(1.12345672), static_cast<float>(1.12345675),
              static_cast<float>(1.12345677), static_cast<float>(1.12345678)},
             {static_cast<float>(1.12345673), static_cast<float>(1.12345676),
              static_cast<float>(1.12345678), static_cast<float>(1.12345679)}}
        )
    );
}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Half) {
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("half_precise.mtx"),
        NoFillMatrixSparse<float>(
            TestBase::bundle,
            {{static_cast<float>(1.120), static_cast<float>(1.124),
              static_cast<float>(1.128), static_cast<float>(1.122)},
             {static_cast<float>(1.121), static_cast<float>(1.125),
              static_cast<float>(1.129), static_cast<float>(1.123)},
             {static_cast<float>(1.122), static_cast<float>(1.126),
              static_cast<float>(1.120), static_cast<float>(1.124)},
             {static_cast<float>(1.123), static_cast<float>(1.127),
              static_cast<float>(1.121), static_cast<float>(1.125)}}
        )
    );
}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Half) {
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("half_precise_symmetric.mtx"),
        NoFillMatrixSparse<float>(
            TestBase::bundle,
            {{static_cast<float>(1.120), static_cast<float>(1.121),
              static_cast<float>(1.122), static_cast<float>(1.123)},
             {static_cast<float>(1.121), static_cast<float>(1.124),
              static_cast<float>(1.125), static_cast<float>(1.126)},
             {static_cast<float>(1.122), static_cast<float>(1.125),
              static_cast<float>(1.127), static_cast<float>(1.128)},
             {static_cast<float>(1.123), static_cast<float>(1.126),
              static_cast<float>(1.128), static_cast<float>(1.129)}}
        )
    );
}