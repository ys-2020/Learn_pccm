import pccm

class check_class(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("cmath")
        self.add_include("iostream")

    # @pccm.pybind.mark
    @pccm.member_function()
    def check(self):
        code = pccm.FunctionCode("")
        code.raw("""
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.f;
                for (int p = 0; p < k; ++p) {
                    sum += A[i * k + p] * B[j + p * n];
                }

                if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                    std::cout << "C[" << i <<"]["<<j<<"] not match, "<< sum << " vs "<< C[i*n+j] << std::endl; 
                    return false;
                }
            }
        }
        return true;
        """).arg("A,B,C","const float *").arg("m,n,k", "int").ret("bool")
        return code