#include "tools/abs.h"

#include <cuda_fp16.h>

#include <cmath>

template <> double cascade::abs_ns::abs(const double &val) {
    return std::abs(val);
}

template <> float cascade::abs_ns::abs(const float &val) {
    return std::abs(val);
}

template <> __half cascade::abs_ns::abs(const __half &val) {
    return __habs(val);
}