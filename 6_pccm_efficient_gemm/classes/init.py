import pccm

class random_init_class(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("cstdlib")

    @pccm.pybind.mark
    @pccm.member_function(const=True, virtual=True)
    def random_init(self):
        code = pccm.FunctionCode("")
        code.raw("""
        for (size_t i = 0; i < size; ++i) {
            data[i] = float(rand()) / RAND_MAX;
        }
        """).arg("data", "float*").arg("size","size_t").ret("void")
        return code

