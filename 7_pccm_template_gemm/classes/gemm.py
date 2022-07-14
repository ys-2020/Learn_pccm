import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
import numpy as np

import pccm
from pccm import builder, core
from pccm.middlewares import pybind
import pickle 

from cumm.common import CUDALibs, GemmBasic, TensorView

from classes.gemm_param import GemmParam
from classes.gemm_param import GLOBAL_GEMM_PARAM

REPO_ROOT = Path("/home/yangshang19/nfs/smr/Learn_pccm")

class GemmDependency(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(CUDALibs)

        self.add_include("mem.cuh")
        self.build_meta.includes.append(REPO_ROOT/"7_pccm_template_gemm"/"include")

class EfficientGemm(pccm.Class):
    def __init__(self):
        super().__init__()
        # self.tmp1 = 1
        # self.tmp2 = 2
        self.add_dependency(GemmDependency)
        self.add_dependency(TensorView)

    @pccm.cuda.cuda_global_function(launch_bounds = [256,2])
    def sgemm_128x128x8_kernel(self):
        code = pccm.FunctionCode("")
        code.ret("void")
        code.arg("A,B","const float *").arg("C","float *")
        code.arg("m,n,k","uint32_t")
        code.arg("A_ldg_step,B_ldg_step","uint32_t")
        # code.arg("c_gemm_param","C_gemm_param")
        code.raw(f"""

                // param passing
                auto smem_switch_a = {GLOBAL_GEMM_PARAM.smem_switch_a};
                auto smem_switch_b = {GLOBAL_GEMM_PARAM.smem_switch_b};
                auto warp_size     = {GLOBAL_GEMM_PARAM.warp_size};
                auto tblock_tile_m = {GLOBAL_GEMM_PARAM.tblock_tile[0]};
                auto tblock_tile_n = {GLOBAL_GEMM_PARAM.tblock_tile[1]};
                auto tblock_tile_k = {GLOBAL_GEMM_PARAM.tblock_tile[2]};
                const int thread_tile_m = {GLOBAL_GEMM_PARAM.thread_tile[0]};
                const int thread_tile_n = {GLOBAL_GEMM_PARAM.thread_tile[1]};
                const int thread_ldg_a  = {GLOBAL_GEMM_PARAM.thread_ldg_a};
                const int thread_ldg_b  = {GLOBAL_GEMM_PARAM.thread_ldg_b};
                auto warp_tile_m   = {GLOBAL_GEMM_PARAM.warp_tile[0]};
                auto warp_tile_n   = {GLOBAL_GEMM_PARAM.warp_tile[1]};
                const int warp_size_m   = {GLOBAL_GEMM_PARAM.warp_size_m};
                const int warp_size_n   = {GLOBAL_GEMM_PARAM.warp_size_n};
                //auto tblock_size_m = {GLOBAL_GEMM_PARAM.tblock_size_m};
                auto tblock_size_n = {GLOBAL_GEMM_PARAM.tblock_size_n};


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
                for (int i = 0; i < thread_tile_m; ++i) {{
                    #pragma unroll
                    for (int j = 0; j < thread_tile_n; ++j) {{
                        C_frag[i][j] = 0;
                    }}
                }}

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
                for (int i = 0; i < thread_ldg_a; ++i) {{
                    /*
                    int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
                    */
                    int m_idx = blockIdx.y * tblock_tile_m + threadIdx.x / tblock_tile_k * thread_ldg_a + i;
                    if (m_idx < m) {{
                        A_ldg_guard |= (1u << i);
                    }}
                }}

                uint32_t B_ldg_guard = 0;
                #pragma unroll
                for (int i = 0; i < thread_ldg_b; ++i) {{
                    /*
                    int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
                    */
                    int n_idx = blockIdx.x * tblock_tile_n + threadIdx.x % (tblock_tile_n / thread_ldg_b) + i * (tblock_tile_n / thread_ldg_b);
                    if (n_idx < n) {{
                        B_ldg_guard |= (1u << i);
                    }}
                }}

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
                {{
                    /*
                    uint32_t first_k_tile = k - k_tiles * 8;
                    */
                    uint32_t first_k_tile = k - k_tiles * tblock_tile_k;

                    #pragma unroll
                    for (int i = 0; i < thread_ldg_a; ++i) {{
                        /*
                        bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                                    threadIdx.x % 8 < first_k_tile;
                        */
                        bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                                    threadIdx.x % tblock_tile_k < first_k_tile;
                        
                        ldg32_nc_0(A_ldg_reg[i],
                                A_ldg_ptr + i * A_ldg_step,
                                guard);
                    }}

                    if constexpr (thread_ldg_a > 4){{
                        #pragma unroll
                        for(int i = 0; i < thread_ldg_a / 4; i++){{
                            sts128(A_ldg_reg[i * 4], A_ldg_reg[i * 4 + 1], A_ldg_reg[i * 4 + 2], A_ldg_reg[i * 4 + 3],
                                A_sts_addr + i * 16);
                        }}
                    }}
                    else if (thread_ldg_a == 4){{
                        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                            A_sts_addr);
                    }}
                    else if (thread_ldg_a == 2){{
                        sts64(A_ldg_reg[0], A_ldg_reg[1], A_sts_addr);
                    }}
                    else{{
                        sts32(A_ldg_reg[0], A_sts_addr);
                    }}

                    #pragma unroll
                    for (int i = 0; i < thread_ldg_b; ++i) {{
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
                    }}

                    #pragma unroll
                    for (int i = 0; i < thread_ldg_b; ++i) {{
                        /*
                        sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                        */
                        sts32(B_ldg_reg[i], B_sts_addr + i * (tblock_tile_n / thread_ldg_b) * sizeof(float));
                    }}

                    __syncthreads();

                    // switch double buffer. TBD: these two numbers are correlated with 128x8 tile size
                    A_sts_addr ^= smem_switch_a;
                    B_sts_addr ^= smem_switch_b;

                    // ldg pointer for next tile
                    A_ldg_ptr += first_k_tile * sizeof(float);
                    B_ldg_ptr += n * first_k_tile * sizeof(float);
                }}

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
                for (; k_tiles > 0; --k_tiles) {{
                    #pragma unroll
                    for(int k_frag = 0; k_frag < tblock_tile_k; ++k_frag){{
                    //for (int k_frag = 0; k_frag < 8; ++k_frag) {{
                        // store next A&B tile to shared memory
                        // Haotian: note: there is one tile not used! so we still need an epilog.
                        if (k_frag == tblock_tile_k - 1){{
                        //if (k_frag == 7) {{
                            if constexpr (thread_ldg_a > 4){{
                                #pragma unroll
                                for(int i = 0; i < thread_ldg_a / 4; i++){{
                                    sts128(A_ldg_reg[i * 4], A_ldg_reg[i * 4 + 1], A_ldg_reg[i * 4 + 2], A_ldg_reg[i * 4 + 3],
                                        A_sts_addr + i * 16);
                                }}
                            }}
                            else if (thread_ldg_a == 4){{
                                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                                    A_sts_addr);
                            }}
                            else if (thread_ldg_a == 2){{
                                sts64(A_ldg_reg[0], A_ldg_reg[1], A_sts_addr);
                            }}
                            else{{
                                sts32(A_ldg_reg[0], A_sts_addr);
                            }}
                            #pragma unroll
                            for (int i = 0; i < thread_ldg_b; ++i) {{
                                /*
                                sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                                */
                                sts32(B_ldg_reg[i], B_sts_addr + i * (tblock_tile_n / thread_ldg_b) * sizeof(float));
                            }}

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
                        }}

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
                        if (k_frag == 0) {{
                            #pragma unroll
                            for (int i = 0; i < thread_ldg_a; ++i) {{
                                ldg32_nc(A_ldg_reg[i],
                                        A_ldg_ptr + i * A_ldg_step,
                                        (A_ldg_guard & (1u << i)) != 0);
                            }}

                            #pragma unroll
                            for (int i = 0; i < thread_ldg_b; ++i) {{
                                /*
                                ldg32_nc(B_ldg_reg[i],
                                        B_ldg_ptr + i * 32 * sizeof(float),
                                        (B_ldg_guard & (1u << i)) != 0);
                                */
                                ldg32_nc(B_ldg_reg[i],
                                        B_ldg_ptr + i * (tblock_tile_n / thread_ldg_b) * sizeof(float),
                                        (B_ldg_guard & (1u << i)) != 0);
                            }}
                        }}

                        // FFMA loop
                        /*
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {{
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {{
                                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                                B_frag[k_frag % 2][j];
                            }}
                        }}
                        */
                        #pragma unroll
                        for (int i = 0; i < thread_tile_m; ++i) {{
                            #pragma unroll
                            for (int j = 0; j < thread_tile_n; ++j) {{
                                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                                B_frag[k_frag % 2][j];
                            }}
                        }}

                    }}
                }}

                // FFMA for the last tile
                #pragma unroll
                //for (int k_frag = 0; k_frag < 8; ++k_frag) {{
                //    if (k_frag < 7) {{
                for (int k_frag = 0; k_frag < tblock_tile_k; ++k_frag) {{
                    if (k_frag < tblock_tile_k - 1) {{
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
                    }}

                    // FFMA loop
                    /*
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {{
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {{
                            C_frag[i][j] += A_frag[k_frag % 2][i] *
                                            B_frag[k_frag % 2][j];
                        }}
                    }}
                    */
                    #pragma unroll
                    for (int i = 0; i < thread_tile_m; ++i) {{
                        #pragma unroll
                        for (int j = 0; j < thread_tile_n; ++j) {{
                            C_frag[i][j] += A_frag[k_frag % 2][i] *
                                            B_frag[k_frag % 2][j];
                        }}
                    }}
                }}

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

                if (m_idx >= m) {{
                    return;
                }} else if (m_idx + warp_tile_m <= m) {{
                    // haotian: 32 = warp_tile_m

                    // haotian: each 4x4 is contiguous. i bound 2 = thread tile / 4, sim. for j bound.
                    // haotian: store to smem contiguously. 
                    // haotian: according to the figure above (reg -> smem transfer). Note: space reused for different i,j.
                    // haotian: smem->gmem: each thread store 1 number, 32 threads access memory contiguously. store 16x32 once.
                    uint32_t n_guard = n < n_idx ? 0 : n - n_idx;
                    /*
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {{
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {{
                            __syncthreads();

                            #pragma unroll
                            for (int p = 0; p < 4; ++p) {{
                                sts128(C_frag[i * 4 + p][j * 4],
                                    C_frag[i * 4 + p][j * 4 + 1],
                                    C_frag[i * 4 + p][j * 4 + 2],
                                    C_frag[i * 4 + p][j * 4 + 3],
                                    C_sts_addr + p * 8 * sizeof(float4));
                            }}

                            __syncthreads();
                            #pragma unroll
                            for (int p = 0; p < 16; ++p) {{
                                stg32(C_lds_ptr[p * 32],
                                    C_stg_ptr + (i * 16 + p) * n + j * 32,
                                    j * 32 < n_guard);
                            }}
                        }}
                    }}
                    */
                    #pragma unroll
                    for (int i = 0; i < thread_tile_m / 4; ++i) {{
                        #pragma unroll
                        for (int j = 0; j < thread_tile_n / 4; ++j) {{
                            __syncthreads();
                            // store a 4x4 for each thread, each warp: 16x32 in default case.
                            #pragma unroll
                            for (int p = 0; p < 4; ++p) {{
                                sts128(C_frag[i * 4 + p][j * 4],
                                    C_frag[i * 4 + p][j * 4 + 1],
                                    C_frag[i * 4 + p][j * 4 + 2],
                                    C_frag[i * 4 + p][j * 4 + 3],
                                    C_sts_addr + p * warp_size_n * sizeof(float4));
                            }}

                            __syncthreads();
                            // haotian: each warp stores 512 = 32 x (4 x 4) elements in one inner-loop step
                            // haotian: currently only solves warp_tile_n > 32, but what about warp_tile_n < 32? q?? (solved)
                            // haotian: to solve slice_k, need to perform reduction here.
                            if constexpr (warp_size_n == 8){{
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {{
                                    stg32(C_lds_ptr[p * 32],
                                        C_stg_ptr + (i * 16 + p) * n + j * 32,
                                        j * 32 < n_guard);
                                }}
                            }}
                            else if (warp_size_n > 8){{
                                #pragma unroll
                                for(int p = 0; p < 4 * warp_size_m; ++p){{
                                    #pragma unroll
                                    for(int q = 0; q < warp_size / 8; ++q){{
                                        stg32(C_lds_ptr[p * 4 * warp_size_n + q * 32],
                                            C_stg_ptr + (i * 4 * warp_size_m + p) * n + j * 4 * warp_size_n + q * 32,
                                            (j * 4 * warp_size_n + q * 32 < n_guard));
                                    }}
                                }}
                            }}
                            else{{
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {{
                                    stg32(C_lds_ptr[p * 32],
                                        C_stg_ptr + (i * 16 + p) * (8 / warp_size_n) * n + j * 4 * warp_size_n,
                                        (j * 4 * warp_size_n < n_guard));
                                }}
                            }}
                        }}
                    }}
                }} /*else {{
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {{
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {{
                            StgFrag stg_frag(C_frag, j, i);

                            C_tile_wb(stg_frag,
                                    C_stg_ptr + i * 16 * n + j * 32,
                                    C_lds_ptr,
                                    C_sts_addr,
                                    m,
                                    n,
                                    m_idx + i * 16,
                                    n_idx + j * 32);
                        }}
                    }}
                }}*/
                else{{
                    uint32_t n_guard = n < n_idx ? 0 : n - n_idx;
                    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;
                    // haotian: just copy the previous code, branch predictor issue
                    #pragma unroll
                    for (int i = 0; i < thread_tile_m / 4; ++i) {{
                        #pragma unroll
                        for (int j = 0; j < thread_tile_n / 4; ++j) {{
                            __syncthreads();
                            #pragma unroll
                            for (int p = 0; p < 4; ++p) {{
                                sts128(C_frag[i * 4 + p][j * 4],
                                    C_frag[i * 4 + p][j * 4 + 1],
                                    C_frag[i * 4 + p][j * 4 + 2],
                                    C_frag[i * 4 + p][j * 4 + 3],
                                    C_sts_addr + p * warp_size_n * sizeof(float4));
                            }}

                            __syncthreads();
                            if constexpr (warp_size_n == 8){{
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {{
                                    stg32(C_lds_ptr[p * 32],
                                        C_stg_ptr + (i * 16 + p) * n + j * 32,
                                        i * 16 + p < m_guard && j * 32 < n_guard);
                                }}
                            }}
                            else if (warp_size_n > 8){{
                                #pragma unroll
                                for(int p = 0; p < 4 * warp_size_m; ++p){{
                                    #pragma unroll
                                    for(int q = 0; q < warp_size / 8; ++q){{
                                        stg32(C_lds_ptr[p * 4 * warp_size_n + q * 32],
                                            C_stg_ptr + (i * 4 * warp_size_m + p) * n + j * 4 * warp_size_n + q * 32,
                                            (i * 4 * warp_size_m + p < m_guard) && (j * 4 * warp_size_n + q * 32 < n_guard));
                                    }}
                                }}
                            }}
                            else{{
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {{
                                    stg32(C_lds_ptr[p * 32],
                                        C_stg_ptr + (i * 16 + p) * (8 / warp_size_n) * n + j * 4 * warp_size_n,
                                        ((i * 16 + p) * (8 / warp_size_n) < m_guard) && (j * 4 * warp_size_n < n_guard));
                                }}
                            }}
                        }}
                    }}
                }}

        """) 
        return code    
