#include "test.h"

#include "tools/read_matrix.h"

class read_vectorMTX_Test: public TestBase
{
public:

    template <typename TPrecision>
    void ReadFirstSparseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector1.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "first"
        ));

        Vector<TPrecision> target(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
             static_cast<TPrecision>(0), static_cast<TPrecision>(0),
             static_cast<TPrecision>(0)}
        );

        ASSERT_VECTOR_NEAR(
            test,
            target,
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void ReadFirstDenseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector2.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "first"
        ));

        Vector<TPrecision> target(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(2),
             static_cast<TPrecision>(3), static_cast<TPrecision>(4),
             static_cast<TPrecision>(5), static_cast<TPrecision>(6)}
        );

        ASSERT_VECTOR_NEAR(
            test,
            target,
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void ReadRandomSparseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector1.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "first"
        ));

        NoFillMatrixSparse<TPrecision> target_mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(2),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(3), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(4)}}
        );

        ASSERT_TRUE(
            (test == target_mat.get_col(0).copy_to_vec()) ||
            (test == target_mat.get_col(1).copy_to_vec()) ||
            (test == target_mat.get_col(2).copy_to_vec()) ||
            (test == target_mat.get_col(3).copy_to_vec())
        );

    }

    template <typename TPrecision>
    void ReadRandomDenseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector2.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "random"
        ));

        NoFillMatrixSparse<TPrecision> target_mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(6),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(5),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(3),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(2),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)}}
        );

        ASSERT_TRUE(
            (test == target_mat.get_col(0).copy_to_vec()) ||
            (test == target_mat.get_col(1).copy_to_vec()) ||
            (test == target_mat.get_col(2).copy_to_vec())
        );

    }

};

TEST_F(read_vectorMTX_Test, ReadFirstSparseVector) {
    ReadFirstSparseVector<double>();
    ReadFirstSparseVector<float>();
    ReadFirstSparseVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadFirstDenseVector) {
    ReadFirstDenseVector<double>();
    ReadFirstDenseVector<float>();
    ReadFirstDenseVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadRandomSparseVector) {
    ReadRandomSparseVector<double>();
    ReadRandomSparseVector<float>();
    ReadRandomSparseVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadRandomDenseVector) {
    ReadRandomDenseVector<double>();
    ReadRandomDenseVector<float>();
    ReadRandomDenseVector<__half>();
}

TEST_F(read_vectorMTX_Test, FailOnBadSelection) {

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            read_vectorMTX<double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("vector1.mtx"),
                "waffles"
            );
        }
    );

}