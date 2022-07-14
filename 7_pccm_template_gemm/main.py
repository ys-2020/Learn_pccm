from classes.wrapper import * 
from classes.gemm_param import GemmParam
from classes.gemm_param import GLOBAL_GEMM_PARAM
    

def test_efficient_gemm(params:GemmParam):
    cu = EfficientGemm_Wrapper(params)
    cu.namespace = "EfficientGemmTest"
    lib = pccm.builder.build_pybind([cu],
                                    Path(__file__).parent / "efficientgemm_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_efficientgemm",
                                    pybind_file_suffix=".cu",   # use nvcc instead of gcc !!
                                    verbose=False,
                                    disable_anno=True,
                                    std="c++17")
    return lib

if __name__ == "__main__":
    GLOBAL_GEMM_PARAM.set_params(M = 256, N =256, K =256)
    # GLOBAL_GEMM_PARAM.show_params()

    lib = test_efficient_gemm(GLOBAL_GEMM_PARAM)

    if (lib.EfficientGemmTest.EfficientGemm_Wrapper.run_efficientgemm() == True):
        print("The result is right! Check Pass!")
    





