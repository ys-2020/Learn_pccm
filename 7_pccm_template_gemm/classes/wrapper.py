import math
from typing import List

import pccm
from pccm.core import FunctionCode
from cumm.common import TensorView as tv

import classes.gemm
from classes.gemm import *
from classes.init import random_init_class
from classes.check import check_class
from classes.gemm_param import GemmParam
from classes.gemm_param import GLOBAL_GEMM_PARAM


class EfficientGemm_Wrapper(pccm.ParameterizedClass):
    def __init__(self,params:GemmParam):
        super().__init__()
        self.add_dependency(random_init_class)
        self.add_dependency(check_class)
        self.add_dependency(EfficientGemm)
        self.params = params

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def run_efficientgemm(self):
        code = pccm.FunctionCode("")
        code.ret("bool")

        code.raw(f"""
        auto Random_Init = random_init_class();
        auto Check_Result = check_class();
        //auto Efficient_Gemm = EfficientGemm();

        // param passing
        int m = {self.params.M};
        int n = {self.params.N};
        int k = {self.params.K};

        auto warp_size     = {self.params.warp_size};
        auto tblock_tile_m = {self.params.tblock_tile[0]};
        auto tblock_tile_n = {self.params.tblock_tile[1]};
        auto tblock_tile_k = {self.params.tblock_tile[2]};
        auto thread_ldg_a  = {self.params.thread_ldg_a};
        auto thread_ldg_b  = {self.params.thread_ldg_b};
        auto warp_size_m   = {self.params.warp_size_m};
        auto warp_size_n   = {self.params.warp_size_n};
        auto tblock_size_m = {self.params.tblock_size_m};
        auto tblock_size_n = {self.params.tblock_size_n};


        int n_iter = 10;
        // 1 warp = 32 threads constraint
        assert(warp_size_m * warp_size_n == warp_size);
        // need to load <= can load. problem why < cannot work: some warps do not load from gmem to smem,
        // but they need to use data from smem in ffma!
        tv::ssprint(
            "A need to load:",tblock_tile_m * tblock_tile_k,
            "; B need to load:",tblock_tile_n * tblock_tile_k,";");
        tv::ssprint(
            "A can load:",thread_ldg_a * warp_size * tblock_size_m * tblock_size_n,
            "; B can load:", thread_ldg_b * warp_size * tblock_size_m * tblock_size_n);
        assert(tblock_tile_m * tblock_tile_k == thread_ldg_a * warp_size * tblock_size_m * tblock_size_n);
        assert(tblock_tile_n * tblock_tile_k == thread_ldg_b * warp_size * tblock_size_m * tblock_size_n);

        float *h_A, *h_B, *h_C;

        cudaMallocHost(&h_A, m * k * sizeof(float));
        cudaMallocHost(&h_B, k * n * sizeof(float));
        cudaMallocHost(&h_C, m * n * sizeof(float));
        Random_Init.random_init(h_A, m * k);
        Random_Init.random_init(h_B, k * n);

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
        
        // warmup
        classes::gemm::sgemm_128x128x8_kernel<<<grid, 32 * tblock_size_m * tblock_size_n>>>(
            d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * tblock_tile_k);

        cudaEventRecord(start);
        for (int i = 0; i < n_iter; ++i) {{
            classes::gemm::sgemm_128x128x8_kernel<<<grid, 32 * tblock_size_m * tblock_size_n>>>(
                d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * tblock_tile_k);
        }}
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float ms;
        cudaEventElapsedTime(&ms, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        long workload = n_iter * long(m) * n * k * 2;
        double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
        tv::ssprint("Performance:",gflops, "GFLOPS");

        cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);

        bool chk = Check_Result.check(h_A, h_B, h_C, m, n, k);
        tv::ssprint("Matrix_C check:", chk ? "OK" : "Failed");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        cudaFreeHost(h_C);

        return chk;

        """)
        return code
