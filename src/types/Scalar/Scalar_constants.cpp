#include <cuda_runtime.h>

#include "types/Scalar/Scalar.h"

template<>
Scalar<__half> SCALAR_ONE<__half>::get() { return SCALAR_ONE_H; }
template<>
Scalar<float> SCALAR_ONE<float>::get() { return SCALAR_ONE_F; }
template<>
Scalar<double> SCALAR_ONE<double>::get() { return SCALAR_ONE_D; }

template<>
Scalar<__half> SCALAR_ZERO<__half>::get() { return SCALAR_ZERO_H; }
template<>
Scalar<float> SCALAR_ZERO<float>::get() { return SCALAR_ZERO_F; }
template<>
Scalar<double> SCALAR_ZERO<double>::get() { return SCALAR_ZERO_D; }

template<>
Scalar<__half> SCALAR_MINUS_ONE<__half>::get() { return SCALAR_MINUS_ONE_H; }
template<>
Scalar<float> SCALAR_MINUS_ONE<float>::get() { return SCALAR_MINUS_ONE_F; }
template<>
Scalar<double> SCALAR_MINUS_ONE<double>::get() { return SCALAR_MINUS_ONE_D; }