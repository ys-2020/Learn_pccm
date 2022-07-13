import pccm

class mem_class(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("cstdint")
        self.add_include("cstdlib")

    # @pccm.pybind.mark
    @pccm.cuda.member_function(device=True,forceinline=True)
    def smem_u32addr(self):
        code = pccm.FunctionCode()
        code.raw("""
            uint32_t addr;
            asm ("{.reg .u64 u64addr;\\n"
            " cvta.to.shared.u64 u64addr, %1;\\n"
            " cvt.u32.u64 %0, u64addr;}\\n"
            : "=r"(addr)
            : "l"(smem_ptr)
        );
        return addr;
        """).arg("smem_ptr", "const void*").ret("uint32_t") 
        return code