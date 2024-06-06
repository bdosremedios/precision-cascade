#include "test.h"

#include "tools/read_matrix.h"

// General matrix read tests
class read_matrixMTX_Test: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void ReadEmptyMatrix() {

        fs::path empty_file(read_matrix_dir / fs::path("empty.mtx"));
        M<double> test_empty(read_matrixMTX<M, double>(TestBase::bundle, empty_file));
        ASSERT_EQ(test_empty.rows(), 0);
        ASSERT_EQ(test_empty.cols(), 0);

    }

    template <template <typename> typename M, typename T>
    void ReadSimpleGeneral() {

        fs::path file_square(read_matrix_dir / fs::path("square.mtx"));
        M<T> target_square(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(4), static_cast<T>(7)},
             {static_cast<T>(2), static_cast<T>(5), static_cast<T>(8)},
             {static_cast<T>(3), static_cast<T>(6), static_cast<T>(9)}}
        );
        M<T> test_square(read_matrixMTX<M, T>(TestBase::bundle, file_square));
        ASSERT_MATRIX_EQ(test_square, target_square);

        fs::path file_tall(read_matrix_dir / fs::path("tall.mtx"));
        M<T> target_tall(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(5)},
             {static_cast<T>(2), static_cast<T>(6)},
             {static_cast<T>(3), static_cast<T>(7)},
             {static_cast<T>(4), static_cast<T>(8)}}
        );
        M<T> test_tall(read_matrixMTX<M, T>(TestBase::bundle, file_tall));
        ASSERT_MATRIX_EQ(test_tall, target_tall);

        fs::path file_wide(read_matrix_dir / fs::path("wide.mtx"));
        M<T> target_wide(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(3), static_cast<T>(5), static_cast<T>(7), static_cast<T>(9)},
             {static_cast<T>(2), static_cast<T>(4), static_cast<T>(6), static_cast<T>(8), static_cast<T>(10)}}
        );
        M<T> test_wide(read_matrixMTX<M, T>(TestBase::bundle, file_wide));
        ASSERT_MATRIX_EQ(test_wide, target_wide);

    }

    template <template <typename> typename M, typename T>
    void ReadSimpleSymmetric() {

        fs::path file_symmetric(read_matrix_dir / fs::path("square_symmetric.mtx"));
        M<T> target_symmetric(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(2), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(3), static_cast<T>(7), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)},
             {static_cast<T>(4), static_cast<T>(8), static_cast<T>(11), static_cast<T>(13), static_cast<T>(14)},
             {static_cast<T>(5), static_cast<T>(9), static_cast<T>(12), static_cast<T>(14), static_cast<T>(15)}}
        );
        M<T> test_symmetric(read_matrixMTX<M, T>(TestBase::bundle, file_symmetric));
        ASSERT_MATRIX_EQ(test_symmetric, target_symmetric);

        fs::path file_gen_match(read_matrix_dir / fs::path("square_general_match.mtx"));
        M<T> target_file_to_match(read_matrixMTX<M, T>(TestBase::bundle, file_symmetric));
        ASSERT_MATRIX_EQ(test_symmetric, target_file_to_match);

    }

    template <template <typename> typename M, typename T>
    void ReadPreciseGeneral(fs::path mtx_path, M<T> target) {

        Scalar<T> permutation(static_cast<T>(1.5)*Tol<T>::roundoff_T());
        M<T> test(read_matrixMTX<M, T>(TestBase::bundle, mtx_path));
        ASSERT_MATRIX_NEAR(test, target, Tol<T>::roundoff_T());
        ASSERT_MATRIX_LT(
            test,
            M<T>(
                MatrixDense<T>(target) +
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*permutation
            )
        );
        ASSERT_MATRIX_GT(
            test,
            M<T>(
                MatrixDense<T>(target) -
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*permutation
            )
        );
        
    }

    template <template <typename> typename M, typename T>
    void ReadPreciseSymmetric(fs::path sym_mtx_path, M<T> target) {

        Scalar<T> permutation(static_cast<T>(1.5)*Tol<T>::roundoff_T());
        M<T> test(read_matrixMTX<M, T>(TestBase::bundle, sym_mtx_path));
        ASSERT_MATRIX_NEAR(test, target, Tol<T>::roundoff_T());
        ASSERT_MATRIX_LT(
            test,
            M<T>(
                MatrixDense<T>(target) +
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*permutation
            )
        );
        ASSERT_MATRIX_GT(
            test,
            M<T>(
                MatrixDense<T>(target) -
                MatrixDense<T>::Ones(TestBase::bundle, target.rows(), target.cols())*permutation
            )
        );
        
    }

    template <template <typename> typename M>
    void BadReadMTX() {

        auto missing_file = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("asdasd.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, missing_file);

        auto wrong_extension = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad1"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, wrong_extension);

        auto bad_header_1 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_header1.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header_1);
        auto bad_header2 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_header2.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header2);
        auto bad_header3 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_header3.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header3);
        auto bad_header4 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_header4.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header4);
        auto bad_header5 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_header5.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header5);
        auto bad_header6 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_header6.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header6);
        auto bad_header7 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_header7.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header7);

        auto bad_matrixdim1 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim1.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim1);
        auto bad_matrixdim2 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim2.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim2);
        auto bad_matrixdim3 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim3.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim3);
        auto bad_matrixdim4 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim4.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim4);
        auto bad_matrixdim5 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim5.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim5);
        auto bad_matrixdim6 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim6.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim6);
        auto bad_matrixdim7 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim7.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim7);
        auto bad_matrixdim8 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixdim8.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim8);

        auto bad_matrixentry1 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry1.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry1);
        auto bad_matrixentry2 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry2.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry2);
        auto bad_matrixentry3 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry3.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry3);
        auto bad_matrixentry4 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry4.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry4);
        auto bad_matrixentry5 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry5.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry5);
        auto bad_matrixentry6 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry6.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry6);
        auto bad_matrixentry7 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry7.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry7);
        auto bad_matrixentry8 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry8.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry8);
        auto bad_matrixentry9 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_matrixentry9.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry9);

        auto bad_wrongorder1 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_wrongorder1.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_wrongorder1);
        auto bad_wrongorder2 = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_wrongorder2.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_wrongorder2);

        auto bad_symmetricinvalidentry = [=]() {
            read_matrixMTX<M, double>(TestBase::bundle, read_matrix_dir / fs::path("bad_symmetricinvalidentry.mtx"));
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_symmetricinvalidentry);

    }

};

TEST_F(read_matrixMTX_Test, ReadEmptyMatrix) {
    ReadEmptyMatrix<MatrixDense, __half>();
    ReadEmptyMatrix<NoFillMatrixSparse, __half>();
    ReadEmptyMatrix<MatrixDense, float>();
    ReadEmptyMatrix<NoFillMatrixSparse, float>();
    ReadEmptyMatrix<MatrixDense, double>();
    ReadEmptyMatrix<NoFillMatrixSparse, double>();
}

TEST_F(read_matrixMTX_Test, ReadSimpleGeneral) {
    ReadSimpleGeneral<MatrixDense, __half>();
    ReadSimpleGeneral<NoFillMatrixSparse, __half>();
    ReadSimpleGeneral<MatrixDense, float>();
    ReadSimpleGeneral<NoFillMatrixSparse, float>();
    ReadSimpleGeneral<MatrixDense, double>();
    ReadSimpleGeneral<NoFillMatrixSparse, double>();
}

TEST_F(read_matrixMTX_Test, ReadSimpleSymmetric) {
    ReadSimpleSymmetric<MatrixDense, __half>();
    ReadSimpleSymmetric<NoFillMatrixSparse, __half>();
    ReadSimpleSymmetric<MatrixDense, float>();
    ReadSimpleSymmetric<NoFillMatrixSparse, float>();
    ReadSimpleSymmetric<MatrixDense, double>();
    ReadSimpleSymmetric<NoFillMatrixSparse, double>();
}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Double) {
    
    MatrixDense<double> mat(
        TestBase::bundle,
        {{static_cast<double>(1.1234567890123450), static_cast<double>(1.1234567890123454),
          static_cast<double>(1.1234567890123458), static_cast<double>(1.1234567890123452)},
         {static_cast<double>(1.1234567890123451), static_cast<double>(1.1234567890123455),
          static_cast<double>(1.1234567890123459), static_cast<double>(1.1234567890123453)},
         {static_cast<double>(1.1234567890123452), static_cast<double>(1.1234567890123456),
          static_cast<double>(1.1234567890123450), static_cast<double>(1.1234567890123454)},
         {static_cast<double>(1.1234567890123453), static_cast<double>(1.1234567890123457),
          static_cast<double>(1.1234567890123451), static_cast<double>(1.1234567890123455)}}
    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("double_precise.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("double_precise.mtx"),
        NoFillMatrixSparse<double>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Double) {

    MatrixDense<double> mat(
        TestBase::bundle,
        {{static_cast<double>(1.1234567890123450), static_cast<double>(1.1234567890123451),
          static_cast<double>(1.1234567890123452), static_cast<double>(1.1234567890123453)},
         {static_cast<double>(1.1234567890123451), static_cast<double>(1.1234567890123454),
          static_cast<double>(1.1234567890123455), static_cast<double>(1.1234567890123456)},
         {static_cast<double>(1.1234567890123452), static_cast<double>(1.1234567890123455),
          static_cast<double>(1.1234567890123457), static_cast<double>(1.1234567890123458)},
         {static_cast<double>(1.1234567890123453), static_cast<double>(1.1234567890123456),
          static_cast<double>(1.1234567890123458), static_cast<double>(1.1234567890123459)}}
    );

    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("double_precise_symmetric.mtx"),
        mat
    );
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("double_precise_symmetric.mtx"),
        NoFillMatrixSparse<double>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Single) {

    MatrixDense<float> mat(
        TestBase::bundle,
        {{static_cast<float>(1.12345670), static_cast<float>(1.12345674),
          static_cast<float>(1.12345678), static_cast<float>(1.12345672)},
         {static_cast<float>(1.12345671), static_cast<float>(1.12345675),
          static_cast<float>(1.12345679), static_cast<float>(1.12345673)},
         {static_cast<float>(1.12345672), static_cast<float>(1.12345676),
          static_cast<float>(1.12345670), static_cast<float>(1.12345674)},
         {static_cast<float>(1.12345673), static_cast<float>(1.12345677),
          static_cast<float>(1.12345671), static_cast<float>(1.12345675)}}

    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("single_precise.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("single_precise.mtx"),
        NoFillMatrixSparse<float>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Single) {
    
    MatrixDense<float> mat(
        TestBase::bundle,
        {{static_cast<float>(1.12345670), static_cast<float>(1.12345671),
          static_cast<float>(1.12345672), static_cast<float>(1.12345673)},
         {static_cast<float>(1.12345671), static_cast<float>(1.12345674),
          static_cast<float>(1.12345675), static_cast<float>(1.12345676)},
         {static_cast<float>(1.12345672), static_cast<float>(1.12345675),
          static_cast<float>(1.12345677), static_cast<float>(1.12345678)},
         {static_cast<float>(1.12345673), static_cast<float>(1.12345676),
          static_cast<float>(1.12345678), static_cast<float>(1.12345679)}}
    );

    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("single_precise_symmetric.mtx"),
        mat
    );
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("single_precise_symmetric.mtx"),
        NoFillMatrixSparse<float>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Half) {

    MatrixDense<__half> mat(
        TestBase::bundle,
        {{static_cast<__half>(1.120), static_cast<__half>(1.124),
          static_cast<__half>(1.128), static_cast<__half>(1.122)},
         {static_cast<__half>(1.121), static_cast<__half>(1.125),
          static_cast<__half>(1.129), static_cast<__half>(1.123)},
         {static_cast<__half>(1.122), static_cast<__half>(1.126),
          static_cast<__half>(1.120), static_cast<__half>(1.124)},
         {static_cast<__half>(1.123), static_cast<__half>(1.127),
          static_cast<__half>(1.121), static_cast<__half>(1.125)}}
    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("half_precise.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("half_precise.mtx"),
        NoFillMatrixSparse<__half>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Half) {

    MatrixDense<__half> mat(
        TestBase::bundle,
        {{static_cast<__half>(1.120), static_cast<__half>(1.121),
          static_cast<__half>(1.122), static_cast<__half>(1.123)},
         {static_cast<__half>(1.121), static_cast<__half>(1.124),
          static_cast<__half>(1.125), static_cast<__half>(1.126)},
         {static_cast<__half>(1.122), static_cast<__half>(1.125),
          static_cast<__half>(1.127), static_cast<__half>(1.128)},
         {static_cast<__half>(1.123), static_cast<__half>(1.126),
          static_cast<__half>(1.128), static_cast<__half>(1.129)}}
    );

    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("half_precise_symmetric.mtx"),
        mat
    );
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("half_precise_symmetric.mtx"),
        NoFillMatrixSparse<__half>(mat)
    );

}

TEST_F(read_matrixMTX_Test, BadReadMTX) {
    BadReadMTX<MatrixDense>();
    BadReadMTX<NoFillMatrixSparse>();
}