#ifndef GENERALMATRIX_GPU_CONSTANTS_CUH
#define GENERALMATRIX_GPU_CONSTANTS_CUH

namespace cascade::genmat_gpu_const
{
    constexpr int WARPSIZE(32);
    constexpr int MAXTHREADSPERBLOCK(1024);
}

#endif