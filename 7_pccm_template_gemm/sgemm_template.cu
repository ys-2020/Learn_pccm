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



void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A,
           const float *B,
           const float *C,
           int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j + p * n];
            }

            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }

    return true;
}

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

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n128k8
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixA: 8x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                128
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |    64    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */

// haotian: seems that thread_tile can only be 8 = 4 (per thread load 4)x 2 (figure above)??

__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel(const float *A,
                            const float *B,
                            float *C,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            uint32_t A_ldg_step,    // k * sizeof(float)
                            uint32_t B_ldg_step) {  // n * sizeof(float) * tblock_tile_k
    /*
     * matrix A & B thread block tile shared memory (double buffer)
     * matrix A: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
     * matrix B: 128 * 8 * 4Byte/item * double buffer = 8KB
     *
     * for double buffer faster switch, A_smem requires 8KB * 2 shared memory
     * and 16KB aligned, B_smem should be 8KB aligned, then the double buffer
     * can be switched by only 1 xor instruction:
     *     (uint32_t &)A_smem ^= 0x2000;
     *     (uint32_t &)B_smem ^= 0x1000;
     */
    
    // [!!] tbd: haotian: check whether the smem space is enough (20220102)
    /*
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);
    */
    __shared__ char smem[24 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);
    
    // 20000 x 32 x 32, 5000 x 32 x 32, 5000 x 128 x 128
    /*
    __shared__ char smem[32 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 28 * 1024);
    */

    // A, B and C register fragment
    float A_frag[2][thread_tile_m];
    float B_frag[2][thread_tile_n];
    float C_frag[thread_tile_m][thread_tile_n];
    #pragma unroll
    for (int i = 0; i < thread_tile_m; ++i) {
        #pragma unroll
        for (int j = 0; j < thread_tile_n; ++j) {
            C_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % warp_size;
    const uint32_t warp_id = threadIdx.x / warp_size;

    // 4x8 threads each warp for FFMA
    
    /*
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    */
    // refer to the warp tile map figure!
    const uint32_t mma_tid_x = (lane_id / 2) % warp_size_n;
    // 2x16: lane_id/32, 4x8: /16, 8x4: /8, 16x2: /4, 32x1: /2
    const uint32_t mma_tid_y = (lane_id / (warp_size_n * 2)) * 2 + (lane_id % 2);
    
    //if(warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0) printf("%d %d %d\n", lane_id, mma_tid_x, mma_tid_y);

    // A_tile & B_tile ldg pointer
    // haotian: [1] equivalent to <<< (m/128, n/128), (32, 8)>>>, **each warp load 4** from DRAM to shared (tblock: 128x8).
    // haotian: [2] equivalent to <<< (m/128, n/128), (8, 32)>>>, **each warp load 4** from DRAM to shared.
    /*
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
    
    const char *B_ldg_ptr = (const char *)(
        B + (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32);
    */
    // haotian: always make sure data loaded from gmem can be directly used (no need to wait for other warps).
    // eg. for A, warp 0 loaded 16x8 at 1st step. Note that later compute will first be using this 16 numbers. (16 x 2) x (32 x 2)
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * tblock_tile_m + threadIdx.x / tblock_tile_k * thread_ldg_a) * k + threadIdx.x % tblock_tile_k);
    
    const char *B_ldg_ptr = (const char *)(
        B + (threadIdx.x / (tblock_tile_n / thread_ldg_b)) * n + blockIdx.x * tblock_tile_n + threadIdx.x % (tblock_tile_n / thread_ldg_b));

    // A_tile & B_tile sts/lds pointer
    // using uint32_t pointer for faster double buffer switch
    // haotian: [1] looks like 8 x 132 shared memory (what is 8x4 for?)
    // haotian: [2] looks like 128 x 8 shared memory
    // haotian: correlate with A_ldg_ptr and B_ldg_ptr, just transposed for A.
    /*
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
    
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
    */
    // haotian: +4 for efficiency. to be analyzed: why?
    uint32_t A_sts_addr = smem_u32addr( 
        A_smem + (threadIdx.x % tblock_tile_k) * (tblock_tile_m + 4) + (threadIdx.x / tblock_tile_k) * thread_ldg_a);
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / (tblock_tile_n / thread_ldg_b)) * tblock_tile_n + (threadIdx.x % (tblock_tile_n / thread_ldg_b)));
    
    // haotian: mma_tid_x in [0, 7] (for B shared mem), mma_tid_y in [0, 3] (for A shared mem), each thread load 4 (strided) x 4 (contig), there might be broadcast.
    // haotian: warp_id is 4x2. per warp 32x1 or 1x64
    /*
    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);
    */
    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / tblock_size_n) * warp_tile_m + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % tblock_size_n) * warp_tile_n + mma_tid_x * 4);

    // ldg_guard to avoid LDG out of bound
    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < thread_ldg_a; ++i) {
        /*
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
        */
        int m_idx = blockIdx.y * tblock_tile_m + threadIdx.x / tblock_tile_k * thread_ldg_a + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }

    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < thread_ldg_b; ++i) {
        /*
        int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
        */
        int n_idx = blockIdx.x * tblock_tile_n + threadIdx.x % (tblock_tile_n / thread_ldg_b) + i * (tblock_tile_n / thread_ldg_b);
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    float A_ldg_reg[thread_ldg_a];
    float B_ldg_reg[thread_ldg_b];

    // 1'st A&B tile loaded before the k_tile loop
    /*
    uint32_t k_tiles = (k + 7) / 8 - 1;
    */
    uint32_t k_tiles = (k + tblock_tile_k - 1) / tblock_tile_k - 1;

    // load 1'st tile to shared memory
    // haotian: A load 128x1, B load 1x128, each thread load 4.
    // a stored as 1x128, B stored as 128x1
    {
        /*
        uint32_t first_k_tile = k - k_tiles * 8;
        */
        uint32_t first_k_tile = k - k_tiles * tblock_tile_k;

        #pragma unroll
        for (int i = 0; i < thread_ldg_a; ++i) {
            /*
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            */
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % tblock_tile_k < first_k_tile;
            
            ldg32_nc_0(A_ldg_reg[i],
                       A_ldg_ptr + i * A_ldg_step,
                       guard);
        }

        if constexpr (thread_ldg_a > 4){
            #pragma unroll
            for(int i = 0; i < thread_ldg_a / 4; i++){
                sts128(A_ldg_reg[i * 4], A_ldg_reg[i * 4 + 1], A_ldg_reg[i * 4 + 2], A_ldg_reg[i * 4 + 3],
                    A_sts_addr + i * 16);
            }
        }
        else if (thread_ldg_a == 4){
            sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                A_sts_addr);
        }
        else if (thread_ldg_a == 2){
            sts64(A_ldg_reg[0], A_ldg_reg[1], A_sts_addr);
        }
        else{
            sts32(A_ldg_reg[0], A_sts_addr);
        }

        #pragma unroll
        for (int i = 0; i < thread_ldg_b; ++i) {
            /*
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 32 < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],    
                       B_ldg_ptr + i * 32 * sizeof(float),
                       guard);
            */
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / (tblock_tile_n / thread_ldg_b) < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],    
                       B_ldg_ptr + i * (tblock_tile_n / thread_ldg_b) * sizeof(float),
                       guard);
        }

        #pragma unroll
        for (int i = 0; i < thread_ldg_b; ++i) {
            /*
            sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
            */
            sts32(B_ldg_reg[i], B_sts_addr + i * (tblock_tile_n / thread_ldg_b) * sizeof(float));
        }

        __syncthreads();

        // switch double buffer. TBD: these two numbers are correlated with 128x8 tile size
        A_sts_addr ^= smem_switch_a;
        B_sts_addr ^= smem_switch_b;

        // ldg pointer for next tile
        A_ldg_ptr += first_k_tile * sizeof(float);
        B_ldg_ptr += n * first_k_tile * sizeof(float);
    }

    // load 1'st fragment
    // haotian: 16 and 32: see the figure above. Assume each thread loads 4.
    /*
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));
    */
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 4 * warp_size_m * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 4 * warp_size_n * sizeof(float));


    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for(int k_frag = 0; k_frag < tblock_tile_k; ++k_frag){
        //for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            // Haotian: note: there is one tile not used! so we still need an epilog.
            if (k_frag == tblock_tile_k - 1){
            //if (k_frag == 7) {
                if constexpr (thread_ldg_a > 4){
                    #pragma unroll
                    for(int i = 0; i < thread_ldg_a / 4; i++){
                        sts128(A_ldg_reg[i * 4], A_ldg_reg[i * 4 + 1], A_ldg_reg[i * 4 + 2], A_ldg_reg[i * 4 + 3],
                            A_sts_addr + i * 16);
                    }
                }
                else if (thread_ldg_a == 4){
                    sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                        A_sts_addr);
                }
                else if (thread_ldg_a == 2){
                    sts64(A_ldg_reg[0], A_ldg_reg[1], A_sts_addr);
                }
                else{
                    sts32(A_ldg_reg[0], A_sts_addr);
                }
                #pragma unroll
                for (int i = 0; i < thread_ldg_b; ++i) {
                    /*
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                    */
                    sts32(B_ldg_reg[i], B_sts_addr + i * (tblock_tile_n / thread_ldg_b) * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_lds_addr ^= smem_switch_a;
                B_lds_addr ^= smem_switch_b;
                A_sts_addr ^= smem_switch_a;
                B_sts_addr ^= smem_switch_b;

                // ldg pointer for next tile
                /*
                A_ldg_ptr += 8 * sizeof(float);
                */
                A_ldg_ptr += tblock_tile_k * sizeof(float);
                // haotian: B_ldg_step = 8 * n * sizeof(float). n dimension tiled over tblock.
                B_ldg_ptr += B_ldg_step;
            }

            // load next A&B fragment from shared memory to register. Haotian: naturally double buffer (k_flag switches between even and odd)
            // +16: each thread load 4 x warp_size_m 4 = 16, +32: each thread load 4 x warp_size_n 8 = 32
            /*
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
            */
            
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % tblock_tile_k * (tblock_tile_m + 4) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % tblock_tile_k * (tblock_tile_m + 4) + 4 * warp_size_m) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % tblock_tile_k * tblock_tile_n * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % tblock_tile_k * tblock_tile_n + 4 * warp_size_n) * sizeof(float));
            
            // load next A&B tile. Haotian: 4 = 128 (tblock tile size) / 32 ??
            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < thread_ldg_a; ++i) {
                    ldg32_nc(A_ldg_reg[i],
                             A_ldg_ptr + i * A_ldg_step,
                             (A_ldg_guard & (1u << i)) != 0);
                }

                #pragma unroll
                for (int i = 0; i < thread_ldg_b; ++i) {
                    /*
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 32 * sizeof(float),
                             (B_ldg_guard & (1u << i)) != 0);
                    */
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * (tblock_tile_n / thread_ldg_b) * sizeof(float),
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            // FFMA loop
            /*
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
            */
            #pragma unroll
            for (int i = 0; i < thread_tile_m; ++i) {
                #pragma unroll
                for (int j = 0; j < thread_tile_n; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }

        }
    }

    // FFMA for the last tile
    #pragma unroll
    //for (int k_frag = 0; k_frag < 8; ++k_frag) {
    //    if (k_frag < 7) {
    for (int k_frag = 0; k_frag < tblock_tile_k; ++k_frag) {
        if (k_frag < tblock_tile_k - 1) {
            // load next A&B fragment from shared memory to register
            /*
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
            */
            
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % tblock_tile_k * (tblock_tile_m + 4) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % tblock_tile_k * (tblock_tile_m + 4) + 4 * warp_size_m) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % tblock_tile_k * tblock_tile_n * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % tblock_tile_k * tblock_tile_n + 4 * warp_size_n) * sizeof(float));
        }

        // FFMA loop
        /*
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
        */
        #pragma unroll
        for (int i = 0; i < thread_tile_m; ++i) {
            #pragma unroll
            for (int j = 0; j < thread_tile_n; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    // haotian: mma_tid_x <= 8, mma_tid_y <= 4, per thread loads 4 data
    
    // [!!] tbd; haotian: in the mma_tid_x direction, since float4, we dont need to x4, but x4 still needed on mma_tid_y direction.
    
    /*
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;
    */
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * (warp_tile_m * warp_tile_n)) +
                                       mma_tid_y * 4 * warp_size_n + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * (warp_tile_m * warp_tile_n)) + lane_id;

    /*
    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;
    */
    uint32_t m_idx = blockIdx.y * tblock_tile_m + warp_id / tblock_size_n * warp_tile_m + lane_id / (4 * warp_size_n);
    uint32_t n_idx = blockIdx.x * tblock_tile_n + warp_id % tblock_size_n * warp_tile_n + lane_id % (4 * warp_size_n);
    // haotian: this might be problematic when warp_tile_n != 64.
    float *C_stg_ptr = C + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else if (m_idx + warp_tile_m <= m) {
        // haotian: 32 = warp_tile_m

        // haotian: each 4x4 is contiguous. i bound 2 = thread tile / 4, sim. for j bound.
        // haotian: store to smem contiguously. 
        // haotian: according to the figure above (reg -> smem transfer). Note: space reused for different i,j.
        // haotian: smem->gmem: each thread store 1 number, 32 threads access memory contiguously. store 16x32 once.
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;
        /*
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 8 * sizeof(float4));
                }

                __syncthreads();
                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 32],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
        */
        #pragma unroll
        for (int i = 0; i < thread_tile_m / 4; ++i) {
            #pragma unroll
            for (int j = 0; j < thread_tile_n / 4; ++j) {
                __syncthreads();
                // store a 4x4 for each thread, each warp: 16x32 in default case.
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * warp_size_n * sizeof(float4));
                }

                __syncthreads();
                // haotian: each warp stores 512 = 32 x (4 x 4) elements in one inner-loop step
                // haotian: currently only solves warp_tile_n > 32, but what about warp_tile_n < 32? q?? (solved)
                // haotian: to solve slice_k, need to perform reduction here.
                if constexpr (warp_size_n == 8){
                    #pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        stg32(C_lds_ptr[p * 32],
                            C_stg_ptr + (i * 16 + p) * n + j * 32,
                            j * 32 < n_guard);
                    }
                }
                else if (warp_size_n > 8){
                    #pragma unroll
                    for(int p = 0; p < 4 * warp_size_m; ++p){
                        #pragma unroll
                        for(int q = 0; q < warp_size / 8; ++q){
                            stg32(C_lds_ptr[p * 4 * warp_size_n + q * 32],
                                C_stg_ptr + (i * 4 * warp_size_m + p) * n + j * 4 * warp_size_n + q * 32,
                                (j * 4 * warp_size_n + q * 32 < n_guard));
                        }
                    }
                }
                else{
                    #pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        //printf("%d %d %d %d\n", (i * 16 + p) * (8 / warp_size_n), j * 4 * warp_size_n, m_idx, n_idx);
                        stg32(C_lds_ptr[p * 32],
                            C_stg_ptr + (i * 16 + p) * (8 / warp_size_n) * n + j * 4 * warp_size_n,
                            (j * 4 * warp_size_n < n_guard));
                    }
                }
            }
        }
    } /*else {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
    }*/
    else{
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;
        uint32_t m_guard = m < m_idx ? 0 : m - m_idx;
        // haotian: just copy the previous code, branch predictor issue
        #pragma unroll
        for (int i = 0; i < thread_tile_m / 4; ++i) {
            #pragma unroll
            for (int j = 0; j < thread_tile_n / 4; ++j) {
                __syncthreads();
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * warp_size_n * sizeof(float4));
                }

                __syncthreads();
                if constexpr (warp_size_n == 8){
                    #pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        stg32(C_lds_ptr[p * 32],
                            C_stg_ptr + (i * 16 + p) * n + j * 32,
                            i * 16 + p < m_guard && j * 32 < n_guard);
                    }
                }
                else if (warp_size_n > 8){
                    #pragma unroll
                    for(int p = 0; p < 4 * warp_size_m; ++p){
                        #pragma unroll
                        for(int q = 0; q < warp_size / 8; ++q){
                            stg32(C_lds_ptr[p * 4 * warp_size_n + q * 32],
                                C_stg_ptr + (i * 4 * warp_size_m + p) * n + j * 4 * warp_size_n + q * 32,
                                (i * 4 * warp_size_m + p < m_guard) && (j * 4 * warp_size_n + q * 32 < n_guard));
                        }
                    }
                }
                else{
                    #pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        //printf("%d %d %d %d\n", (i * 16 + p) * (8 / warp_size_n), j * 4 * warp_size_n, m_idx, n_idx);
                        stg32(C_lds_ptr[p * 32],
                            C_stg_ptr + (i * 16 + p) * (8 / warp_size_n) * n + j * 4 * warp_size_n,
                            ((i * 16 + p) * (8 / warp_size_n) < m_guard) && (j * 4 * warp_size_n < n_guard));
                    }
                }
            }
        }
    }

    //if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) printf("%f %f %f %f\n", C_frag[0][0], C_frag[1][0], C_frag[2][0], C_frag[3][0]);
    //if(threadIdx.x == 15 && blockIdx.x == 0 && blockIdx.y == 0) printf("%f %f %f %f %f\n", smem[0], smem[16], smem[32], smem[48], smem[64]);
}

int main() {
    /*
    int m = 5120;
    int n = 4096;
    int k = 4096;
    */
    /*
    int m = 20000;
    int n = 32;
    int k = 32;
    */
    int m = 5120;
    int n = 4096;
    int k = 4096;
    /*
    int m = 5000;
    int n = 128;
    int k = 128;
    */
    int n_iter = 10;
    // 1 warp = 32 threads constraint
    assert(warp_size_m * warp_size_n == warp_size);
    // need to load <= can load. problem why < cannot work: some warps do not load from gmem to smem,
    // but they need to use data from smem in ffma!
    printf(
        "A need to load: %d; B need to load: %d;\nA can load: %d; B can load: %d.\n", 
        tblock_tile_m * tblock_tile_k, tblock_tile_n * tblock_tile_k, thread_ldg_a * warp_size * tblock_size_m * tblock_size_n, thread_ldg_b * warp_size * tblock_size_m * tblock_size_n
    );
    assert(tblock_tile_m * tblock_tile_k == thread_ldg_a * warp_size * tblock_size_m * tblock_size_n);
    assert(tblock_tile_n * tblock_tile_k == thread_ldg_b * warp_size * tblock_size_m * tblock_size_n);

    float *h_A, *h_B, *h_C;

    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid((n + tblock_tile_n - 1) / tblock_tile_n, (m + tblock_tile_m - 1) / tblock_tile_m);
    
    //printf("%d %d %d\n", (n + tblock_tile_n - 1) / tblock_tile_n, (m + tblock_tile_m - 1) / tblock_tile_m, 32 * tblock_size_m * tblock_size_n);
    
    // warmup
    sgemm_128x128x8_kernel<<<grid, 32 * tblock_size_m * tblock_size_n>>>(
        d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * tblock_tile_k);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        sgemm_128x128x8_kernel<<<grid, 32 * tblock_size_m * tblock_size_n>>>(
            d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * tblock_tile_k);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);

    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}

