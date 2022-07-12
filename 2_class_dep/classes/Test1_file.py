import pccm
import torch


class Test1(pccm.Class):
    def __init__(self):
        super().__init__()  # init function of pccm.Class do NOT have any arguments 
        self.add_include("iostream","cmath")
    
    # @pccm.pybind.mark(nogil=True)
    @pccm.member_function(inline=True)  # Note that we are using member_func here
    def add(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return a + b;
        """).arg("a,b", "int").ret("int")
        return code
