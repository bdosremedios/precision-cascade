#include "test.h"

#include "types/Scalar/Scalar.h"

class Scalar_Test: public TestBase
{
public:

    template <typename TPrecision>
    void TestGetSet() {

        Scalar<TPrecision> scalar_1;
        Scalar<TPrecision> scalar_2;

        scalar_1.set_scalar(static_cast<TPrecision>(5.));
        scalar_2.set_scalar(static_cast<TPrecision>(-3.1));

        ASSERT_EQ(scalar_1.get_scalar(), static_cast<TPrecision>(5.));
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<TPrecision>(-3.1));

        scalar_1.set_scalar(static_cast<TPrecision>(0.));
        scalar_2.set_scalar(static_cast<TPrecision>(5.));

        ASSERT_EQ(scalar_1.get_scalar(), static_cast<TPrecision>(0.));
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<TPrecision>(5.));

    }

    template <typename TPrecision>
    void TestConstruction() {

        Scalar<TPrecision> scalar_1;
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(5.));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(-1.2));
        Scalar<TPrecision> scalar_4(static_cast<TPrecision>(0.));

        ASSERT_EQ(scalar_2.get_scalar(), static_cast<TPrecision>(5.));
        ASSERT_EQ(scalar_3.get_scalar(), static_cast<TPrecision>(-1.2));
        ASSERT_EQ(scalar_4.get_scalar(), static_cast<TPrecision>(0.));
    }

    template <typename TPrecision>
    void TestCopyAssignment() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(5.));
        scalar_1 = Scalar<TPrecision>(static_cast<TPrecision>(-1.2));
        ASSERT_EQ(scalar_1.get_scalar(), static_cast<TPrecision>(-1.2));
        scalar_1 = Scalar<TPrecision>(static_cast<TPrecision>(0.));
        ASSERT_EQ(scalar_1.get_scalar(), static_cast<TPrecision>(0.));

    }

    template <typename TPrecision>
    void TestCopyConstruction() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(5.));
        Scalar<TPrecision> scalar_2(scalar_1);
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<TPrecision>(5.));

        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(-42.5));
        Scalar<TPrecision> scalar_4(scalar_3);
        ASSERT_EQ(scalar_4.get_scalar(), static_cast<TPrecision>(-42.5));

    }

    template <typename TPrecision>
    void TestAddSub() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(3.2));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(5.));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(-4.));

        ASSERT_NEAR(
            (scalar_1+scalar_2).get_scalar(),
            static_cast<TPrecision>(8.2),
            (min_1_mag(static_cast<TPrecision>(8.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1+scalar_3).get_scalar(),
            static_cast<TPrecision>(-0.8),
            (min_1_mag(static_cast<TPrecision>(-0.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_2+scalar_3).get_scalar(),
            static_cast<TPrecision>(1.),
            (min_1_mag(static_cast<TPrecision>(1.)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1+scalar_1).get_scalar(),
            static_cast<TPrecision>(6.4),
            (min_1_mag(static_cast<TPrecision>(6.4)) *
             Tol<TPrecision>::roundoff_T())
        );
        
        ASSERT_NEAR(
            (scalar_2+scalar_1).get_scalar(),
            static_cast<TPrecision>(8.2),
            (min_1_mag(static_cast<TPrecision>(8.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_3+scalar_1).get_scalar(),
            static_cast<TPrecision>(-0.8),
            (min_1_mag(static_cast<TPrecision>(-0.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_3+scalar_2).get_scalar(),
            static_cast<TPrecision>(1.),
            (min_1_mag(static_cast<TPrecision>(1.)) *
             Tol<TPrecision>::roundoff_T())
        );

        ASSERT_NEAR(
            (scalar_1-scalar_2).get_scalar(),
            static_cast<TPrecision>(-1.8),
            (min_1_mag(static_cast<TPrecision>(-1.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1-scalar_3).get_scalar(),
            static_cast<TPrecision>(7.2),
            (min_1_mag(static_cast<TPrecision>(7.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_2-scalar_3).get_scalar(),
            static_cast<TPrecision>(9.),
            (min_1_mag(static_cast<TPrecision>(9.)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1-scalar_1).get_scalar(),
            static_cast<TPrecision>(0.),
            (min_1_mag(static_cast<TPrecision>(0.)) *
             Tol<TPrecision>::roundoff_T())
        );
        
        ASSERT_NEAR(
            (scalar_2-scalar_1).get_scalar(),
            static_cast<TPrecision>(1.8),
            (min_1_mag(static_cast<TPrecision>(1.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_3-scalar_1).get_scalar(),
            static_cast<TPrecision>(-7.2),
            (min_1_mag(static_cast<TPrecision>(-7.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_3-scalar_2).get_scalar(),
            static_cast<TPrecision>(-9.),
            (min_1_mag(static_cast<TPrecision>(-9.)) *
             Tol<TPrecision>::roundoff_T())
        );

    }
    
    template <typename TPrecision>
    void TestAddSubAssignment() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(3.2));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(5.));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(-4.));

        scalar_2 += scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<TPrecision>(8.2),
            (min_1_mag(static_cast<TPrecision>(8.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_3 += scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(-0.8),
            (min_1_mag(static_cast<TPrecision>(-0.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_3 += scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(2.4),
            (min_1_mag(static_cast<TPrecision>(2.4)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_1 += scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<TPrecision>(6.4),
            (min_1_mag(static_cast<TPrecision>(6.4)) *
             Tol<TPrecision>::roundoff_T())
        );

        scalar_1.set_scalar(static_cast<TPrecision>(3.2));
        scalar_2.set_scalar(static_cast<TPrecision>(5.));
        scalar_3.set_scalar(static_cast<TPrecision>(-4.));
        scalar_2 -= scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<TPrecision>(1.8),
            (min_1_mag(static_cast<TPrecision>(1.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_3 -= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(-7.2),
            (min_1_mag(static_cast<TPrecision>(-7.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_3 -= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(-10.4),
            (min_1_mag(static_cast<TPrecision>(-10.4)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_1 -= scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<TPrecision>(0.),
            (min_1_mag(static_cast<TPrecision>(0.)) *
             Tol<TPrecision>::roundoff_T())
        );

    }

    template <typename TPrecision>
    void TestMultDiv() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(-3.));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(2.4));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(5.));
        Scalar<TPrecision> scalar_4(static_cast<TPrecision>(0.));

        ASSERT_NEAR(
            (scalar_1*scalar_1).get_scalar(),
            static_cast<TPrecision>(9.),
            (min_1_mag(static_cast<TPrecision>(9.)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1*scalar_2).get_scalar(),
            static_cast<TPrecision>(-7.2),
            (min_1_mag(static_cast<TPrecision>(-7.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1*scalar_3).get_scalar(),
            static_cast<TPrecision>(-15.),
            (min_1_mag(static_cast<TPrecision>(-15.)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1*scalar_4).get_scalar(),
            static_cast<TPrecision>(0.),
            (min_1_mag(static_cast<TPrecision>(0.)) *
             Tol<TPrecision>::roundoff_T())
        );

        ASSERT_NEAR(
            (scalar_2*scalar_1).get_scalar(),
            static_cast<TPrecision>(-7.2),
            (min_1_mag(static_cast<TPrecision>(-7.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_2*scalar_2).get_scalar(),
            static_cast<TPrecision>(5.76),
            (min_1_mag(static_cast<TPrecision>(5.76)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_2*scalar_3).get_scalar(),
            static_cast<TPrecision>(12.),
            (min_1_mag(static_cast<TPrecision>(12.)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_2*scalar_4).get_scalar(),
            static_cast<TPrecision>(0.),
            (min_1_mag(static_cast<TPrecision>(0.)) *
             Tol<TPrecision>::roundoff_T())
        );

        ASSERT_NEAR(
            (scalar_1/scalar_1).get_scalar(),
            static_cast<TPrecision>(1.),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(1.)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1/scalar_2).get_scalar(),
            static_cast<TPrecision>(-1.25),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(-1.25)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_1/scalar_3).get_scalar(),
            static_cast<TPrecision>(-3./5.),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(-3./5.)) *
             Tol<TPrecision>::roundoff_T())
        );

        ASSERT_NEAR(
            (scalar_2/scalar_1).get_scalar(),
            static_cast<TPrecision>(-0.8),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(-0.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_2/scalar_2).get_scalar(),
            static_cast<TPrecision>(1.),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(1.)) *
             Tol<TPrecision>::roundoff_T())
        );
        ASSERT_NEAR(
            (scalar_2/scalar_3).get_scalar(),
            static_cast<TPrecision>(0.48),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(0.48)) *
             Tol<TPrecision>::roundoff_T())
        );

    }

    template <typename TPrecision>
    void TestMultDivAssignment() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(-3.));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(2.4));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(5.));
        Scalar<TPrecision> scalar_4(static_cast<TPrecision>(0.));

        scalar_2 *= scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<TPrecision>(-7.2),
            (min_1_mag(static_cast<TPrecision>(-7.2)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_3 *= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(-15.),
            (min_1_mag(static_cast<TPrecision>(-15.)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_4 *= scalar_1;
        ASSERT_NEAR(
            scalar_4.get_scalar(),
            static_cast<TPrecision>(0.),
            (min_1_mag(static_cast<TPrecision>(0.)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_1 *= scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<TPrecision>(9.),
            (min_1_mag(static_cast<TPrecision>(9.)) *
             Tol<TPrecision>::roundoff_T())
        );

        scalar_1.set_scalar(static_cast<TPrecision>(-3.));
        scalar_2.set_scalar(static_cast<TPrecision>(2.4));
        scalar_3.set_scalar(static_cast<TPrecision>(5.));

        scalar_2 /= scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<TPrecision>(-0.8),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(-0.8)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_3 /= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(5./-3.),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(5./-3.)) *
             Tol<TPrecision>::roundoff_T())
        );
        scalar_1 /= scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<TPrecision>(1.),
            (static_cast<TPrecision>(2) *
             min_1_mag(static_cast<TPrecision>(1.)) *
             Tol<TPrecision>::roundoff_T())
        );

    }

    template <typename TPrecision>
    void TestBooleanEqual() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(7.1));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(-7.1));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(7.1));
        Scalar<TPrecision> scalar_4(static_cast<TPrecision>(3.2));

        ASSERT_TRUE(scalar_1 == scalar_1);
        ASSERT_TRUE(scalar_1 == scalar_3);
        ASSERT_TRUE(scalar_3 == scalar_1);

        ASSERT_FALSE(scalar_1 == scalar_2);
        ASSERT_FALSE(scalar_2 == scalar_1);
        ASSERT_FALSE(scalar_1 == scalar_4);
        ASSERT_FALSE(scalar_4 == scalar_1);

    }

    template <typename TPrecision>
    void TestAbs() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(4.2));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(-5.6));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(0.));
        Scalar<TPrecision> scalar_4(static_cast<TPrecision>(-100.));

        ASSERT_EQ(scalar_1.abs().get_scalar(), static_cast<TPrecision>(4.2));
        ASSERT_EQ(scalar_2.abs().get_scalar(), static_cast<TPrecision>(5.6));
        ASSERT_EQ(scalar_3.abs().get_scalar(), static_cast<TPrecision>(0.));
        ASSERT_EQ(scalar_4.abs().get_scalar(), static_cast<TPrecision>(100.));

        ASSERT_EQ(scalar_1.get_scalar(), static_cast<TPrecision>(4.2));
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<TPrecision>(5.6));
        ASSERT_EQ(scalar_3.get_scalar(), static_cast<TPrecision>(0.));
        ASSERT_EQ(scalar_4.get_scalar(), static_cast<TPrecision>(100.));

    }

    template <typename TPrecision>
    void TestSqrt() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(9.));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(0.64));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(100.));

        ASSERT_NEAR(
            scalar_1.sqrt().get_scalar(),
            static_cast<TPrecision>(3.),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_2.sqrt().get_scalar(),
            static_cast<TPrecision>(0.8),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_3.sqrt().get_scalar(),
            static_cast<TPrecision>(10.),
            Tol<TPrecision>::roundoff_T()
        );

        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<TPrecision>(3.),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<TPrecision>(0.8),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(10.),
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void TestReciprocol() {

        Scalar<TPrecision> scalar_1(static_cast<TPrecision>(9.));
        Scalar<TPrecision> scalar_2(static_cast<TPrecision>(0.64));
        Scalar<TPrecision> scalar_3(static_cast<TPrecision>(100.));

        ASSERT_NEAR(
            scalar_1.reciprocol().get_scalar(),
            static_cast<TPrecision>(1./9.),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_2.reciprocol().get_scalar(),
            static_cast<TPrecision>(1./.64),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_3.reciprocol().get_scalar(),
            static_cast<TPrecision>(1./100.),
            Tol<TPrecision>::roundoff_T()
        );

        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<TPrecision>(1./9.),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<TPrecision>(1./.64),
            Tol<TPrecision>::roundoff_T()
        );
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<TPrecision>(1./100.),
            Tol<TPrecision>::roundoff_T()
        );

    }

    void TestCast() {

        Scalar<__half> scalar_half(static_cast<__half>(3.2));
        Scalar<__half> half_to_half(scalar_half.cast<__half>());
        Scalar<float> half_to_float(scalar_half.cast<float>());
        Scalar<double> half_to_double(scalar_half.cast<double>());

        ASSERT_EQ(scalar_half, half_to_half);
        ASSERT_NEAR(
            half_to_float.get_scalar(),
            static_cast<float>(3.2),
            (min_1_mag(static_cast<float>(3.2)) *
             static_cast<float>(Tol<half>::roundoff_T()))
        );
        ASSERT_NEAR(
            half_to_double.get_scalar(),
            static_cast<double>(3.2),
            (min_1_mag(static_cast<double>(3.2)) *
             static_cast<double>(Tol<half>::roundoff_T()))
        );

        Scalar<float> scalar_float(static_cast<float>(-40.6));
        Scalar<__half> float_to_half(scalar_float.cast<__half>());
        Scalar<float> float_to_float(scalar_float.cast<float>());
        Scalar<double> float_to_double(scalar_float.cast<double>());

        ASSERT_EQ(scalar_float, float_to_float);
        ASSERT_NEAR(
            float_to_half.get_scalar(),
            static_cast<__half>(-40.6),
            (min_1_mag(static_cast<__half>(-40.6)) *
             Tol<__half>::roundoff_T())
        );
        ASSERT_NEAR(
            float_to_double.get_scalar(),
            static_cast<double>(-40.6),
            (min_1_mag(static_cast<double>(-40.6)) *
             static_cast<double>(Tol<float>::roundoff_T()))
        );

        Scalar<double> scalar_double(static_cast<double>(2.6));
        Scalar<__half> double_to_half(scalar_double.cast<__half>());
        Scalar<float> double_to_float(scalar_double.cast<float>());
        Scalar<double> double_to_double(scalar_double.cast<double>());

        ASSERT_EQ(scalar_double, double_to_double);
        ASSERT_NEAR(
            double_to_half.get_scalar(),
            static_cast<__half>(2.6),
            (min_1_mag(static_cast<__half>(2.6))*Tol<__half>::roundoff_T())
        );
        ASSERT_NEAR(
            double_to_float.get_scalar(),
            static_cast<float>(2.6),
            (min_1_mag(static_cast<float>(2.6))*Tol<float>::roundoff_T())
        );

    }

    void TestBadCast() {

        auto try_scalar_bad_cast = []() {
            // Scalar<double> scalar(static_cast<double>(0.));
            // scalar.cast<int>();
            throw std::runtime_error("");
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_scalar_bad_cast);

    }

};

TEST_F(Scalar_Test, TestGetSet) {
    TestGetSet<__half>();
    TestGetSet<float>();
    TestGetSet<double>();
}

TEST_F(Scalar_Test, TestConstruction) {
    TestConstruction<__half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(Scalar_Test, TestCopyAssignment) {
    TestCopyAssignment<__half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(Scalar_Test, TestCopyConstruction) {
    TestCopyConstruction<__half>();
    TestCopyConstruction<float>();
    TestCopyConstruction<double>();
}

TEST_F(Scalar_Test, TestAddSub) {
    TestAddSub<__half>();
    TestAddSub<float>();
    TestAddSub<double>();
}

TEST_F(Scalar_Test, TestAddSubAssignment) {
    TestAddSubAssignment<__half>();
    TestAddSubAssignment<float>();
    TestAddSubAssignment<double>();
}

TEST_F(Scalar_Test, TestMultDiv) {
    TestMultDiv<__half>();
    TestMultDiv<float>();
    TestMultDiv<double>();
}

TEST_F(Scalar_Test, TestMultDivAssignment) {
    TestMultDivAssignment<__half>();
    TestMultDivAssignment<float>();
    TestMultDivAssignment<double>();
}

TEST_F(Scalar_Test, TestBooleanEqual) {
    TestBooleanEqual<__half>();
    TestBooleanEqual<float>();
    TestBooleanEqual<double>();
}

TEST_F(Scalar_Test, TestAbs) {
    TestAbs<__half>();
    TestAbs<float>();
    TestAbs<double>();
}

TEST_F(Scalar_Test, TestSqrt) {
    TestSqrt<__half>();
    TestSqrt<float>();
    TestSqrt<double>();
}

TEST_F(Scalar_Test, TestReciprocol) {
    TestReciprocol<__half>();
    TestReciprocol<float>();
    TestReciprocol<double>();
}

TEST_F(Scalar_Test, TestCast) { TestCast(); }

TEST_F(Scalar_Test, TestBadCast) { TestBadCast(); }