#include "types/Scalar/Scalar.h"

#include <cuda_runtime.h>

template<>
cascade::Scalar<__half> cascade::SCALAR_ONE<__half>::get() {
    return SCALAR_ONE_H;
}
template<>
cascade::Scalar<float> cascade::SCALAR_ONE<float>::get() {
    return SCALAR_ONE_F;
}
template<>
cascade::Scalar<double> cascade::SCALAR_ONE<double>::get() {
    return SCALAR_ONE_D;
}

template<>
cascade::Scalar<__half> cascade::SCALAR_ZERO<__half>::get() {
    return SCALAR_ZERO_H;
}
template<>
cascade::Scalar<float> cascade::SCALAR_ZERO<float>::get() {
    return SCALAR_ZERO_F;
}
template<>
cascade::Scalar<double> cascade::SCALAR_ZERO<double>::get() {
    return SCALAR_ZERO_D;
}

template<>
cascade::Scalar<__half> cascade::SCALAR_MINUS_ONE<__half>::get() {
    return SCALAR_MINUS_ONE_H;
}
template<>
cascade::Scalar<float> cascade::SCALAR_MINUS_ONE<float>::get() {
    return SCALAR_MINUS_ONE_F;
}
template<>
cascade::Scalar<double> cascade::SCALAR_MINUS_ONE<double>::get() {
    return SCALAR_MINUS_ONE_D;
}