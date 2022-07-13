#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

#define a_const 132

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

// Check the Results with naive implementation.
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

// ys: convert smem_ptr to unsigned 32 int for faster double buffer switch
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

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
        #pragma unroll  // ys: unroll the loops
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
__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel(const float *A,
                            const float *B,
                            float *C,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            uint32_t A_ldg_step,    // k * sizeof(float)
                            uint32_t B_ldg_step) {  // n * sizeof(float) * 8
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
    __shared__ __align__(16 * 1024) char smem[24 * 1024];    //ys: Align smem to 16K
    float *A_smem = reinterpret_cast<float *>(smem);         //ys: re-interpret content of smem in "float" pattern 
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024); //ys: Shift 16KB

    // A, B and C register fragment
    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }
    // ys: // Clear the thread_tile: Task for a single thread

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;

    // 4x8 threads each warp for FFMA
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    
    /* ys:
     * Threads in a warp would be like:
     *
     * 0  1  16 17
     * 2  3  18 19
     * 4  5  20 21
     * 6  7  22 23
     * 8  9  24 25
     * 10 11 26 27
     * 12 13 28 29
     * 14 15 30 31
     * 
     * But why?
     */


    // A_tile & B_tile ldg pointer
    // haotian: equivalent to <<< (m/128, n/128), (32, 8)>>>, each warp load 4 from DRAM to shared (tblock: 128x8).
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
    
    // ys:
    // 1 block —— 128 A rows.
    // We have 8 cols in each K_tile, so (threadIdx.x % 8) is the col_idx of current thread
    // 8 threads will take 4 rows, so each thread is dealing with 4 elements in A.
    
    // haotian: equivalent to <<< (m/128, n/128), (8, 32)>>>, each warp load 4 from DRAM to shared.
    const char *B_ldg_ptr = (const char *)(
        B + (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32);

    // ys:
    // 1 block —— 128 B cols
    // 32 threads —— 1 B row 
    // **  (K_tiles = 8, and we have 256 threads, so 32 threads is assigned to each row of B_tile)
    // in each B row, there are 128 elements and 32 threads, so each thread is dealing with 4 elements as well.


    // ys: NOT UNDERSTAND!! Why 132 for A?

    // A_tile & B_tile sts/lds pointer   (sts:store to smem  lds:load from smem)
    // using uint32_t pointer for faster double buffer switch
    // haotian: looks like 8 x 132 shared memory (what is 8x4 for?)
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * a_const + (threadIdx.x / 8) * 4);
    // haotian: looks like 128 x 8 shared memory
    // ys: each thread get 4 Bytes(one uint32) of Smem.
    // ys: data in Mat A will be broadcast to 32 threads in a warp.

    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
    // haotian: mma_tid_x in [0, 7] (for B shared mem), 
    //          mma_tid_y in [0, 3] (for A shared mem), 
    //          each thread load 4 (strided) x 4 (contig), 
    //          there might be broadcast.
    
    // ys: 32 threads get 128 Bytes, each thread get 4 Byte for B
    // ys: But why each thread's addr only +1?
    // ys: what is strided and contig?
    
    // haotian: warp_id is 4x2. per warp 32x1 or 1x64
    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);


    // ldg_guard to avoid LDG out of bound
    
    // ys: EACH THREAD HAS ITS OWN LDG_GUARD!!!!
    // ys: In the last 8 threads, they may deal with less than 4 rows, we need to record the num of rows.
    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }

    // ys: In the last block, there might be less than 128 cols of B, 
    //     and (threadIdx.x % 32 + i * 32) is the col that thread will deal with.
    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    float A_ldg_reg[4];
    float B_ldg_reg[4];

    // 1'st A&B tile loaded before the k_tile loop
    uint32_t k_tiles = (k + 7) / 8 - 1; // ys: k is given (A:M*K), not a iterator.

    // load 1'st tile to shared memory. 
    // haotian: A load 128x1, B load 1x128, each thread load 4.
    // A stored as 1x128, B stored as 128x1
    // ys: The 1st tile is not complete (maybe less than 8 cols for A)
    {
        uint32_t first_k_tile = k - k_tiles * 8; // ys: first_k_tile ranges from 1 to 8

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            ldg32_nc_0(A_ldg_reg[i],
                       A_ldg_ptr + i * A_ldg_step,  // A_ldg_step =  k * sizeof(float);  jump to the next row of A
                       guard);
        }

        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
               A_sts_addr);
        // ys: store continuous 128 bits to smem

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 32 < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],    
                       B_ldg_ptr + i * 32 * sizeof(float),
                       guard);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
        }
        // ys: store 4*32 bits to smem

        __syncthreads();

        // switch double buffer
        A_sts_addr ^= 0x2000;   // ys: add/minus 8KB
        B_sts_addr ^= 0x1000;   // ys: add/minus 4KB
        // A_sts_addr is shifted from A_smem (No more than 4KB)
        // B_sts_addr is shifted from B_smem (No more than 4KB)
        // there is 16KB from A_smem to B_smem
        // Thus, it will work.
        // (A needs 8KB because the author is computing with 132*8*4 = 4.125KB)
 

        // ldg pointer for next tile
        A_ldg_ptr += first_k_tile * sizeof(float);
        B_ldg_ptr += n * first_k_tile * sizeof(float);
        // shift the ptr so that it will be the multiple of 8
    }

    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));

    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            // Haotian: note: there is one tile not used! so we still need an epilog.
            if (k_frag == 7) {
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                       A_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x1000;
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x1000;

                // ldg pointer for next tile
                A_ldg_ptr += 8 * sizeof(float);
                B_ldg_ptr += B_ldg_step; // B_ldg_step = n * sizeof(float) * 8; jump 8 rows 
            }

            // load next A&B fragment from shared memory to register. Haotian: naturally double buffer (k_flag switches between even and odd)
            // +16: totally load 32 on A_frag dimension (load 64 on B_frag dimension). Half is 16 or 32.
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * a_const * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * a_const + 16) * sizeof(float));
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

            // load next A&B tile. Haotian: 4 = 128 (tblock tile size) / 32 ??
            // ys: I agree. Since we have 256 threads, and K_tile = 8,
            //              4 = 128 / (256/8)
            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(A_ldg_reg[i],
                             A_ldg_ptr + i * A_ldg_step,
                             (A_ldg_guard & (1u << i)) != 0);
                }

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 32 * sizeof(float),
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            // FFMA loop (Outer Product)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * a_const * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * a_const + 16) * sizeof(float));
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
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }


    // C_tile write back, reuse A&B tile shared memory buffer
    // haotian: mma_tid_x <= 8, mma_tid_y <= 4, per thread loads 4 data
    // ys: float4 combines 4 floats. 
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx;
    
    if (m_idx >= m) {
        return;
    } else if (m_idx + 32 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

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
    } else {
        /*
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
        */
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;
        uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

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
                          i * 16 < m_guard && j * 32 < n_guard);
                }
            }
        }
    }
}

int main() {
    
    int m = 5120;
    int n = 4096;
    int k = 4096;
    
    /*
    int m = 5000;
    int n = 128;
    int k = 128;
    */
    int n_iter = 10;

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

    dim3 grid((n + 127) / 128, (m + 127) / 128);

    // warmup
    sgemm_128x128x8_kernel<<<grid, 256>>>(
        d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        sgemm_128x128x8_kernel<<<grid, 256>>>(
            d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);
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


