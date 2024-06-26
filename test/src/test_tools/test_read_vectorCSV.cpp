#include "test.h"

#include "tools/read_matrix.h"

class read_vectorCSV_Test: public TestBase
{
public:

    template <typename TPrecision>
    void ReadVector() {

        Vector<TPrecision> target(TestBase::bundle, {1, 2, 3, 4, 5, 6});

        fs::path vector_file(read_matrix_dir / fs::path("vector.csv"));
        Vector<TPrecision> test(read_vectorCSV<TPrecision>(
            TestBase::bundle, vector_file
        ));

        ASSERT_VECTOR_NEAR(
            test,
            target,
            static_cast<TPrecision>(Tol<TPrecision>::roundoff())
        );

    }

};

TEST_F(read_vectorCSV_Test, ReadDoubleVector) { ReadVector<double>(); }
TEST_F(read_vectorCSV_Test, ReadSingleVector) { ReadVector<float>(); }
TEST_F(read_vectorCSV_Test, ReadHalfVector) { ReadVector<__half>(); }

TEST_F(read_vectorCSV_Test, FailOnMatrix) {    
    fs::path mat(read_matrix_dir / fs::path("square1.csv"));
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [=]() {
            Vector<double> test(read_vectorCSV<double>(
                TestBase::bundle, mat
            ));
        }
    );
}