#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include "assert.h"

// implicit constraint:
// tblock_tile_m * tblock_tile_k = 4 * warp_size * tblock_size_m * tblock_size_n


#define smem_switch_a 0x2000
#define smem_switch_b 0x1000

// seems that this have to be 2^k??
// 20000 x 32 x 32
/*
#define smem_switch_a (16 * 1024)
#define smem_switch_b 1024
*/

#define warp_size 32
// default values: 128, 128, 8. to try 64, 128, 4


// 20000 x 32 x 32
/*
#define tblock_tile_m 256
#define tblock_tile_n 32
#define tblock_tile_k 8
*/
// 5000 x 32 x 32
/*
#define tblock_tile_m 128
#define tblock_tile_n 32
#define tblock_tile_k 4
*/
// 5000 x 128 x 128
/*
#define tblock_tile_m 128
#define tblock_tile_n 128
#define tblock_tile_k 4
*/
#define tblock_tile_m 128
#define tblock_tile_n 128
#define tblock_tile_k 8

/*
#define warp_tile_m 32
#define warp_tile_n 64
*/
#define thread_tile_m 8
#define thread_tile_n 8
// 20000 x 32 x 32
/*
#define thread_ldg_a 16
#define thread_ldg_b 2
*/
// 5000 x 32 x 32
/*
#define thread_ldg_a 8
#define thread_ldg_b 2
*/
// 5000 x 128 x 128
/*
#define thread_ldg_a 2
#define thread_ldg_b 2
*/
#define thread_ldg_a 4
#define thread_ldg_b 4

// 5000/20000 x 32 x 32
/*
#define warp_tile_m 64
#define warp_tile_n 32
*/
// 5000 x 128 x 128
#define warp_tile_m 32
#define warp_tile_n 64

#define warp_size_m (warp_tile_m/thread_tile_m)
#define warp_size_n (warp_tile_n/thread_tile_n)
#define tblock_size_m (tblock_tile_m/warp_tile_m)
#define tblock_size_n (tblock_tile_n/warp_tile_n)

// warp_tile_m / thread_tile_m
/*
#define warp_size_m 4
// warp_tile_n / thread_tile_n
#define warp_size_n 8
// tblock_tile_m / warp_tile_m
#define tblock_size_m 4
// tblock_tile_n / warp_tile_n
#define tblock_size_n 2
*/

/*
#define warp_tile_m 64
#define warp_tile_n 32
#define warp_size_m 8
#define warp_size_n 4
#define tblock_size_m 2
#define tblock_size_n 4
*/

/*
#define warp_tile_m 128
#define warp_tile_n 16
#define warp_size_m 16
#define warp_size_n 2
#define tblock_size_m 1
#define tblock_size_n 8
*/

__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void stg32(const float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

__device__ __forceinline__
void sts64(const float &reg0, const float &reg1,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v2.f32 [%0], {%1, %2};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

// __device__ __noinline__
// void C_tile_wb(StgFrag C_frag,
//                float *C_stg_ptr,
//                const float *C_lds_ptr,
//                uint32_t C_sts_addr,
//                uint32_t m,
//                uint32_t n,
//                uint32_t m_idx,
//                uint32_t n_idx) {
//     __syncthreads();

//     #pragma unroll
//     for (int i = 0; i < 4; ++i) {
//         sts128(C_frag.data[i][0],
//                C_frag.data[i][1],
//                C_frag.data[i][2],
//                C_frag.data[i][3],
//                C_sts_addr + i * 8 * sizeof(float4));
//     }

//     __syncthreads();

//     uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

//     #pragma unroll
//     for (int i = 0; i < 16; ++i) {
//         stg32(C_lds_ptr[i * 32],
//               C_stg_ptr + i * n,
//               i < m_guard && n_idx < n);
//     }
// }