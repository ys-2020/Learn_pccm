# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

from cumm.gemm.constants import NVRTCConstants

os.environ["CUMM_DEBUG"] = "1"
# _cudart = ctypes.CDLL('libcudart.so')

import pickle
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pccm
import torch
from pccm.core import CodeFormatter

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.gemm import kernel
from cumm.gemm.algospec.core import ShuffleStrideType
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.main import GemmMainUnitTest, NVRTCMode, gen_gemm_kernels

# def cu_prof_start():
#     ret = _cudart.cudaProfilerStart()
#     if ret != 0:
#         raise Exception('cudaProfilerStart() returned %d' % ret)

# def cu_prof_stop():
#     ret = _cudart.cudaProfilerStop()
#     if ret != 0:
#         raise Exception('cudaProfilerStop() returned %d' % ret)

_SPCONV_ROOT = Path(__file__).parent.parent.parent
_GEMM_ROOT = _SPCONV_ROOT / "src/spgemm/gemmdev"
_REFGEMM_ROOT = _SPCONV_ROOT / "src/spgemm/refops"


def build_gemm_lib(cus: List[pccm.Class]):
    lib = pccm.builder.build_pybind(cus,
                                    Path(__file__).parent / "mygemm_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_unittest",
                                    pybind_file_suffix=".cc",
                                    verbose=False,
                                    disable_anno=True,
                                    global_header_only=False,
                                    std="c++17")

    return lib

COUNTBIT_LOOKUP_TABLE = np.array([bin(i).count('1')
                                  for i in range(256)]).astype(np.int32)


from cumm.nvrtc import CummNVRTCModule


def _asdv_test_regular_gemm():
    np.random.seed(12315)
    lib_object = None 
    use_nvrtc = True 
    with cudasim.enter_debug_context(True, 3):
        main_cu = GemmMainUnitTest()
        main_cu.namespace = "cumm.gemm.main"

    if not use_nvrtc:
        lib = build_gemm_lib([main_cu])
        lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = tv.gemm.GemmParams
    algo_cls = tv.gemm.GemmAlgoDesp
    nvrtc_mode = NVRTCMode.ConstantMemory
    a = tv.zeros([3], tv.int32, 0)
    for params in main_cu.all_params:
        if params.shuffle_stride != ShuffleStrideType.NoShuffle:
            continue
        ker = gen_gemm_kernels(params, nvrtc_mode=nvrtc_mode)
        ker.namespace = "test_lib"
        t = time.time()
        custom_names = []
        if nvrtc_mode == NVRTCMode.ConstantMemory:
            custom_names = [f"&test_lib::{NVRTCConstants.CONSTANT_PARAM_KEY}"]

        mod = CummNVRTCModule(
            [ker],
            cudadevrt_path="/usr/local/cuda/lib64/libcudadevrt.a",
            verbose=False,
            custom_names=custom_names)
        # print(mod.get_ptx())

        mod.load()
        print(mod.kernels)
        print("RTC COMPILE TIME", time.time() - t)
        # print(mod.kernels)
        # breakpoint()
        m = 256 + 32
        n = 256 + 40
        k = 136
        m *= 2
        n *= 2
        k *= 2
        m = 64
        n = 64
        k = 64
        m = max(params.ts[0], m)
        n = max(params.ts[1], n)
        k = max(params.ts[2], k)
        if params.dtype_a == dtypes.int8:
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
            dtype_c = params.dtype_c.npdtype()
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
        else:
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            # a[:, 32:] = 0
            # b[32:] = 0
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
        # print("DATA GEN FINISH")
        if params.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if params.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if params.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
        # print("test_lib PREPARED")
        if params.splitk_serial:
            ksplit = 16
        else:
            ksplit = 1
        # print("CUDA PREPARED")
        algo = algo_cls()
        algo.tile_shape = params.ts
        algo.warp_tile_shape = params.wts
        algo.num_stage = params.num_stage
        algo.dacc = params.dtype_acc.tv_dtype
        algo.dcomp = params.dtype_comp.tv_dtype
        algo.algo = params.algo.value
        algo.trans_a = params.trans_a
        algo.trans_b = params.trans_b
        algo.trans_c = params.trans_c
        if params.tensorop is not None:
            algo.tensorop = params.tensorop.shape
        params_cpp = params_cls()
        params_cpp.algo_desp = algo
        params_cpp.split_k_slices = ksplit
        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()

        c_tv = tv.zeros(c.shape, params.dtype_c.tv_dtype, 0)

        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv

        nvrtc_params = tv.gemm.NVRTCParams()
        nvrtc_params.cumodule = mod.get_cpp_object()
        nvrtc_params.mode = nvrtc_mode.value
        nvrtc_params.num_threads = ker.num_threads
        nvrtc_params.smem_size = ker.smem_size
        if nvrtc_mode == NVRTCMode.DynamicParallism:
            nvrtc_params.kernel_name = mod.get_lowered_name(
                "test_lib::nvrtc_kernel")

        elif nvrtc_mode == NVRTCMode.KernelAndCPU:
            nvrtc_params.kernel_name = mod.get_lowered_name("test_lib::gemm_kernel")
            nvrtc_params.init_kernel_name = mod.get_lowered_name(
                "test_lib::nvrtc_kernel_cpu_out")
            nvrtc_params.param_size = mod.const_values[
                f"test_lib::{NVRTCConstants.SIZEOF_KEY}"]

            nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                                  tv.uint8, 0)
            nvrtc_params.param_storage_cpu = tv.empty(
                [nvrtc_params.param_size], tv.uint8, -1, pinned=True)

        elif nvrtc_mode == NVRTCMode.Direct:
            nvrtc_params.kernel_name = mod.get_lowered_name("test_lib::gemm_kernel")
        elif nvrtc_mode == NVRTCMode.ConstantMemory:
            nvrtc_params.kernel_name = mod.get_lowered_name("test_lib::gemm_kernel")
            nvrtc_params.init_kernel_name = mod.get_lowered_name(
                "test_lib::nvrtc_kernel_cpu_out")
            nvrtc_params.param_size = mod.const_values[
                f"test_lib::{NVRTCConstants.SIZEOF_KEY}"]
            nvrtc_params.constant_name = mod.get_lowered_name(
                f"&test_lib::{NVRTCConstants.CONSTANT_PARAM_KEY}")
            nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                                  tv.uint8, 0)
        else:
            raise NotImplementedError
        
        if lib_object is not None:
            lib_object.matmul2(params_cpp)
        else:
            params_cpp.nvrtc_params = nvrtc_params
            with tv.measure_and_print():
                tv.gemm.run_nvrtc_gemm_kernel(params_cpp)
        c_cpu = c_tv.cpu().numpy()
        print(c_cpu.reshape(-1)[-16:])
        print(c.reshape(-1)[-16:])
        
        print(params_cpp.algo_desp, a.mean(), b.mean(), c.mean(),
              np.linalg.norm(c_cpu - c))

if __name__ == "__main__":
    _asdv_test_regular_gemm()
