import pccm
import torch

class cuda_add_class(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("iostream","cmath")

    # @pccm.pybind.mark
    @pccm.cuda.member_function()  
    def cuda_add(self):
        code = pccm.FunctionCode("")
        code.raw("""
        *c = a + b;
        """).arg("a,b", "int").arg("c","int *").ret("void")
        return code

