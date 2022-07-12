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

import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

# import codeai.visualization as vis
import numpy as np

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.gemm import kernel
from cumm.gemm.main import GemmMainUnitTest, gen_gemm_kernels

import os

os.environ["CUMM_DEBUG"] = "1"


def transpose(l):
    return list(map(list, zip(*l)))


def offset_to_coord(offset: np.ndarray, stride: int) -> np.ndarray:
    return np.stack([offset // stride, offset % stride], axis=1)


def _asdv_test_simt_python(coord_input: bool = False):
    with cudasim.enter_debug_context(True):

        main_cu = GemmMainUnitTest()
        for params in main_cu.simt_params[:1]:
            np.random.seed(12315)

            ker = gen_gemm_kernels(params)
            if params.algo != kernel.GemmAlgo.SimtDP4A:
                m = 256 + 32
                n = 256 + 40
                k = 24
                m = 64
                n = 64
                k = 16
                # m = max(params.ts[0], m)
                # n = max(params.ts[1], n)
                # k = max(params.ts[2], k)

                a = np.random.uniform(-1, 1, size=[m, k]).astype(
                    dtypes.get_npdtype(params.dtype_a))
                b = np.random.uniform(-1, 1, size=[k, n]).astype(
                    dtypes.get_npdtype(params.dtype_b))

                c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
                c_dev_0 = a[:4] @ b[:, :4]
                c_dev_1 = a[16:20] @ b[:, :4]
                c_dev_2 = a[4:8] @ b[:, :4]

                acc_dev = np.concatenate([c_dev_0, c_dev_1])
                print(acc_dev.shape, acc_dev.mean(), acc_dev.max(),
                      acc_dev.min())
                print(np.concatenate([a[:4, :2], a[16:20, :2]]))
                c_dev_0 = a[:4, :2] @ b[:2, :4]
                c_dev_1 = a[16:20, :2] @ b[:2, :4]

                acc_dev = np.concatenate([c_dev_0, c_dev_1])
                print(acc_dev.shape, acc_dev.mean(), acc_dev.max(),
                      acc_dev.min())

            else:
                m = 256 + 32
                n = 256 + 40
                k = 56
                m = 64
                n = 128
                k = 32
                m = max(params.ts[0], m)
                n = max(params.ts[1], n)
                k = max(params.ts[2], k)
                print(m, n, k)
                a = np.random.randint(-5, 5, size=[m, k]).astype(np.int8)
                b = np.random.randint(-5, 5, size=[k, n]).astype(np.int8)
                # print("DATA GEN FINISH")
                dtype_np_c = dtypes.get_npdtype(params.dtype_c)
                c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                    dtypes.get_npdtype(params.dtype_c))
                c_dev_0 = a[:4] @ b[:, :4]
                c_dev_1 = a[16:20] @ b[:, :4]
                c_dev_2 = a[4:8] @ b[:, :4]
                print(params.trans_a, params.trans_b, params.trans_c)
                print(c_dev_0)
                print(a.T[:, :4])
                print(b[:, :8])

            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))
            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)

            if coord_input:
                if params.trans_a:
                    for i in range(k):
                        for j in range(m):
                            a_meta[i, j] = (i * m + j)
                else:
                    for i in range(m):
                        for j in range(k):
                            a_meta[i, j] = (i * k + j)
                if params.trans_b:
                    for i in range(n):
                        for j in range(k):
                            b_meta[i, j] = (i * k + j)
                else:
                    for i in range(k):
                        for j in range(n):
                            b_meta[i, j] = (i * n + j)

            a_tv = a.copy()
            b_tv = b.copy()
            c_tv = np.zeros_like(c)
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0])
            duration = time.time() - t

            vis_res = {}
            for k, v in vis_res_per_thread.items():
                for k2, v2 in v.items():
                    if k2 not in vis_res:
                        vis_res[k2] = {}
                    vis_res[k2][k] = v2

            vis_in_relay(list(fig_per_group.values()))
            # print(TestCase().assertAllClose(c_tv, c))
            print(c_tv.reshape(-1)[:10], c.reshape(-1)[:10])
            print(c_tv.reshape(-1)[-10:], c.reshape(-1)[-10:])

            print(params.algo, a.mean(), b.mean(),
                  np.linalg.norm(c_tv - c), "Time=", duration)


def _asdv_test_volta_python(coord_input: bool):

    np.random.seed(12315)
    with cudasim.enter_debug_context(True):
        main_cu = GemmMainUnitTest()
        for params in main_cu.volta_params[:1]:
            ker = gen_gemm_kernels(params)
            m = 256 + 32
            n = 256 + 40
            k = 32
            m = 64
            n = 64
            k = 32
            m = max(params.ts[0], m)
            n = max(params.ts[1], n)
            k = max(params.ts[2], k)

            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))

            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)

            if coord_input:
                if params.trans_a:
                    for i in range(k):
                        for j in range(m):
                            a_meta[i, j] = (i * m + j)
                else:
                    for i in range(m):
                        for j in range(k):
                            a_meta[i, j] = (i * k + j)
                if params.trans_b:
                    for i in range(n):
                        for j in range(k):
                            b_meta[i, j] = (i * k + j)
                else:
                    for i in range(k):
                        for j in range(n):
                            b_meta[i, j] = (i * n + j)
            a_tv = a.copy()
            b_tv = b.copy()
            c_tv = np.zeros_like(c)
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0])
            duration = time.time() - t

            # print(TestCase().assertAllClose(c_tv, c))
            # print(c_tv.reshape(-1)[:10] -  c.reshape(-1)[:10])
            # print(c_tv.reshape(-1)[-10:] -  c.reshape(-1)[-10:])

            print(params.algo, a.mean(), np.linalg.norm(c_tv - c),
                  "Time=", duration)


def unittest_python():
    np.random.seed(12315)
    with cudasim.enter_debug_context(False):
        main_cu = GemmMainUnitTest()
        for params in main_cu.all_params:
            t = time.time()
            ker = gen_gemm_kernels(params)
            m = params.ts[0]
            n = params.ts[1]
            k = params.ts[2]

            if params.dtype_a == dtypes.int8:
                a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
                b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
                dtype_c = params.dtype_c.npdtype()
                c = (a.astype(dtype_c) @ b.astype(dtype_c)).astype(
                    dtypes.get_npdtype(params.dtype_c))

            else:
                a = np.random.uniform(-1, 1, size=[m, k]).astype(
                    dtypes.get_npdtype(params.dtype_a))
                b = np.random.uniform(-1, 1, size=[k, n]).astype(
                    dtypes.get_npdtype(params.dtype_b))
                c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))
            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)
            a_tv = a.copy()
            b_tv = b.copy()
            c_tv = np.zeros_like(c)
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0])
            duration = time.time() - t
            print(params.algo, a.mean(), np.linalg.norm(c_tv - c),
                  "Time=", duration)


def _asdv_test_turing_python(coord_input: bool = False):
    np.random.seed(12315)
    with cudasim.enter_debug_context(True, 3):
        main_cu = GemmMainUnitTest()
        print("len(main_cu.all_params)=",len(main_cu.all_params))

        for params in main_cu.all_params[:1]:
            print(params.algo)

            ker = gen_gemm_kernels(params)

            # print("START", params.algo)
            m = 256 + 32
            n = 256 + 40
            k = 32

            
            m = 32
            n = 32
            k = 32
            m = max(params.ts[0], m)
            n = max(params.ts[1], n)
            k = max(params.ts[2], k)
            print(m, n, k)
            
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

                c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
            if params.trans_a:
                a = np.ascontiguousarray(a.transpose(1, 0))
            if params.trans_b:
                b = np.ascontiguousarray(b.transpose(1, 0))
            if params.trans_c:
                c = np.ascontiguousarray(c.transpose(1, 0))
            # print("WTF PREPARED")
            a_meta = np.zeros_like(a, dtype=np.int64)
            b_meta = np.zeros_like(b, dtype=np.int64)

            if coord_input:
                if params.trans_a:
                    for i in range(k):
                        for j in range(m):
                            a_meta[i, j] = (i * m + j)
                else:
                    for i in range(m):
                        for j in range(k):
                            a_meta[i, j] = (i * k + j)
                if params.trans_b:
                    for i in range(n):
                        for j in range(k):
                            b_meta[i, j] = (i * k + j)
                else:
                    for i in range(k):
                        for j in range(n):
                            b_meta[i, j] = (i * n + j)

            a_tv = a.copy()
            b_tv = b.copy()
            cc_tv = np.zeros_like(c)

            c_tv = np.zeros_like(c)
            # asyncio.run(cudasim.kernel_launch)
            t = time.time()
            vis_res_per_thread, blocks, threads = main_cu.matmul_python(
                a_tv,
                b_tv,
                c_tv,
                a_meta,
                b_meta,
                params.trans_a,
                params.trans_b,
                params.trans_c,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc,
                dcomp=params.dtype_comp,
                algo=params.algo.value,
                tensorop=[0, 0, 0],
                split_k_slices=1)
            duration = time.time() - t


            # print(TestCase().assertAllClose(c_tv, c))
            # print(c_tv.reshape(-1)[:10], c.reshape(-1)[:10])
            # print(c_tv.reshape(-1)[-10:] -  c.reshape(-1)[-10:])

            print(params.algo, a.mean(), b.mean(), c.mean(),
                  np.linalg.norm(c_tv - c), "Time=", duration)

            exit()
            # vis_in_relay(list(fig_per_group.values()))


if __name__ == "__main__":
    # fig = vis.figure.PointCloudFigure(0, np.zeros((1, 3)))
    # with fig.layer("WTF") as layer:
    #     coords = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=np.float32)

    #     layer.add_object(GridPlane([0, 0, 128, 8], [0.5, 0.5], 'green'))
    #     layer.add_object(Coords(coords, [0.5, 0.5], 4, 'red'))

    # unittest_python()
    # _asdv_test_simt_python(True)
    _asdv_test_turing_python(True)
    # _asdv_test_volta_python(True)
