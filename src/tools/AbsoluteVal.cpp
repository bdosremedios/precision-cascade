#include "tools/AbsoluteVal.h"

#include <cmath>
#include <cuda_fp16.h>

template <> double abs_ns::abs(const double &val) {
    return std::abs(val);
}

template <> float abs_ns::abs(const float &val) {
    return std::abs(val);
}

template <> __half abs_ns::abs(const __half &val) {
    return __habs(val);
}