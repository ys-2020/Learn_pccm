import math
from typing import List

import pccm
from pccm.core import FunctionCode

from cumm.common import TensorView as tv
from classes.gemm import *
from classes.init import random_init_class
from classes.check import check_class
import classes.gemm


class EfficientGemm_Wrapper(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(random_init_class)
        self.add_dependency(check_class)
        self.add_dependency(EfficientGemm)

    
    # @pccm.pybind.mark()
    @pccm.cuda.static_function
    def run_efficientgemm(self):
        code = pccm.FunctionCode("")
        code.ret("bool")

        code.raw(f"""
        auto Random_Init = random_init_class();
        auto Check_Result = check_class();
        auto Efficient_Gemm = EfficientGemm();

        int m = 5120;
        int n = 4096;
        int k = 4096;

        int n_iter = 10;

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

        dim3 grid((n + 127) / 128, (m + 127) / 128);

        // warmup

        classes::gemm::sgemm_128x128x8_kernel<<<grid,256>>>(
        d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);

        cudaEventRecord(start);
        for (int i = 0; i < n_iter; ++i) {{
        classes::gemm::sgemm_128x128x8_kernel<<<grid,256>>>(
                d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);
        }}
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float ms;
        cudaEventElapsedTime(&ms, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        long workload = n_iter * long(m) * n * k * 2;
        double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);

        cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);

        bool chk = Check_Result.check(h_A, h_B, h_C, m, n, k);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        cudaFreeHost(h_C);

        return chk;

        """)
        return code