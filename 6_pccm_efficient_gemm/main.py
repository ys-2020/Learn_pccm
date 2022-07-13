from classes.wrapper import *
    

def test_efficient_gemm():
    cu = EfficientGemm_Wrapper()
    cu.namespace = "EfficientGemmTest"
    lib = pccm.builder.build_pybind([cu],
                                    Path(__file__).parent / "efficientgemm_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_efficientgemm",
                                    pybind_file_suffix=".cc",
                                    verbose=False,
                                    disable_anno=True,
                                    std="c++17")
    return lib

if __name__ == "__main__":
    lib = test_efficient_gemm()

    # print(lib.EfficientGemmTest.EfficientGemm_Wrapper())
    # lib.classes.wrapper.EfficientGemm_Wrapper.run_efficientgemm()
    print("Check Pass!")



# def build_efficient_gemm(cu: EfficientGemm):
#     # params = cu.params
#     # cu.namespace = "EfficientGemmTest"
#     lib = pccm.builder.build_pybind([cu],
#                                     Path(__file__).parent / "efficientgemm_test",
#                                     # build_dir=Path(__file__).parent / "build" /
#                                     # "build_efficientgemm",
#                                     # pybind_file_suffix=".cc",
#                                     # verbose=False,
#                                     # disable_anno=True,
#                                     std="c++17")
#     return lib

# def test_efficient_gemm():
#     # params = EfficientGemmParams() #原来用了GemmAlgoParams,是现成的参数类
#     main_cu = EfficientGemm()   
#     lib = build_efficient_gemm(main_cu)
#     return lib






