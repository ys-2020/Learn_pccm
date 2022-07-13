from classes.wrapper import * 
    

def test_efficient_gemm():
    cu = EfficientGemm_Wrapper()
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
    lib = test_efficient_gemm()
    if (lib.EfficientGemmTest.EfficientGemm_Wrapper.run_efficientgemm() == True):
        print("The result is right! Check Pass!")






