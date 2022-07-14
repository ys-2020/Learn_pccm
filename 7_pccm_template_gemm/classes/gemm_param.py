class GemmParam():
    def __init__(self,
                 M = 5120, N = 4096, K = 4096,
                 smem_switch_a=0x2000,smem_switch_b=0x1000,
                 warp_size=32,
                 tblock_tile = [128,128,8],
                 thread_tile = [8,8],
                 warp_tile = [32,64],
                 thread_ldg_a=4,thread_ldg_b=4
                 ):
        self.M = M
        self.N = N
        self.K = K
        self.smem_switch_a = smem_switch_a
        self.smem_switch_b = smem_switch_b
        self.warp_size = warp_size
        self.tblock_tile = tblock_tile
        self.thread_tile = thread_tile
        self.warp_tile = warp_tile
        self.thread_ldg_a = thread_ldg_a
        self.thread_ldg_b = thread_ldg_b

        self.warp_size_m =  int(self.warp_tile[0]/self.thread_tile[0])
        self.warp_size_n =  int(self.warp_tile[1]/self.thread_tile[1])
        self.tblock_size_m = int(self.tblock_tile[0]/self.warp_tile[0])
        self.tblock_size_n = int(self.tblock_tile[1]/self.warp_tile[1])


    def set_params(self,
                 M = 5120, N = 4096, K = 4096,
                 smem_switch_a=0x2000,smem_switch_b=0x1000,
                 warp_size=32,
                 tblock_tile = [128,128,8],
                 thread_tile = [8,8],
                 warp_tile = [32,64],
                 thread_ldg_a=4,thread_ldg_b=4):
        self.M = M
        self.N = N
        self.K = K
        self.smem_switch_a = smem_switch_a
        self.smem_switch_b = smem_switch_b
        self.warp_size = warp_size
        self.tblock_tile = tblock_tile
        self.thread_tile = thread_tile
        self.warp_tile = warp_tile
        self.thread_ldg_a = thread_ldg_a
        self.thread_ldg_b = thread_ldg_b

        self.warp_size_m =  int(self.warp_tile[0]/self.thread_tile[0])
        self.warp_size_n =  int(self.warp_tile[1]/self.thread_tile[1])
        self.tblock_size_m = int(self.tblock_tile[0]/self.warp_tile[0])
        self.tblock_size_n = int(self.tblock_tile[1]/self.warp_tile[1])

    def show_params(self):
        print("M = ",self.M)
        print("N = ",self.N)
        print("K = ",self.K)
        print("smem_switch_a = ",hex(self.smem_switch_a))
        print("smem_switch_b = ",hex(self.smem_switch_b))
        print("warp_size = ",self.warp_size)
        print("warp_size_m = ",self.warp_size_m)
        print("warp_size_n = ",self.warp_size_n)
        print("tblock_tile = ",self.tblock_tile)
        print("thread_tile = ",self.thread_tile)
        print("thread_ldg_a = ",self.thread_ldg_a)
        print("thread_ldg_b = ",self.thread_ldg_b)
        print("tblock_size_m = ",self.tblock_size_m)
        print("tblock_size_n = ",self.tblock_size_n)



GLOBAL_GEMM_PARAM =  GemmParam()