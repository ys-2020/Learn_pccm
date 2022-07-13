import pccm
from .Test1_file import Test1


class Test2(pccm.Class):
    def __init__(self):
        super().__init__()  # init function of pccm.Class do NOT have any arguments 
        self.add_dependency(Test1)

    # @pccm.pybind.mark(nogil=True)
    @pccm.static_function(inline=True)
    def test2_cal(self):    # compute (a+b)*c-d
        code = pccm.FunctionCode("")
        code.raw("""
        auto test1 = Test1();
        auto a_plus_b = test1.add(a, b); 
        return a_plus_b*c-d;
        """).arg("a,b,c,d", "int").ret("int")
        return code



class Test3(Test2):     # inherit from class Test2. We can call func in Test2 by Test3.
    def __init__(self):
        super().__init__()
        self.add_include("iostream")

    @pccm.pybind.mark  
    @pccm.static_function   # use static_func for outside calling
    def test3_sum_of_4(self):    # compute (a+b+c+d)
        code = pccm.FunctionCode("")
        code.raw("""
        return a+b+c+d;
        """).arg("a,b,c,d", "int").ret("int")
        return code

