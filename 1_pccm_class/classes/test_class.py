import pccm
import torch



class Test1(pccm.Class):
    def __init__(self):
        super().__init__()  # init function of pccm.Class do NOT have any arguments 
        # self.add_dependency(torch.Tensor.view) # add dependency for Test1
        self.add_include("iostream","cmath")
        # self.add_member("add_func","std::function<std::int,")
        # self.add_typedef("value_type","char")
        self.add_static_const("kConstVal","int","5")
        # self.add_enum_class("Mode",[("kConvolution",0),
        #                             ("kCrossCorrelation",1)])
    
    # @pccm.pybind.mark(nogil=True)
    # @pccm.member_function(inline=True)
    @pccm.pybind.mark(nogil=True)
    @pccm.static_function(inline=True)
    def add(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return a + b + kConstVal;
        """).arg("a", "int").arg("b","double").ret("int")   # multiple types of variables.  
        return code
