from pccm import builder, core
import pickle 
from pccm.middlewares import pybind
from pathlib import Path
import pccm

from classes.cuda_add import cuda_add_class
from classes import kernel
from cumm.common import TensorView



class cuda_add_wrapper(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(cuda_add_class)
        self.add_dependency(TensorView)
        self.add_kernel = gen_add_kernel()

    # @pccm.pybind.mark
    # @pccm.cuda.static_function
    # def cuda_add_run(self):
    #     code = pccm.FunctionCode("")
    #     code.raw("""
    #      //auto CUDA_ADD = cuda_add_class();

    #      int c;
    #      int *dev_c;
    #      cudaMalloc( (void**)&dev_c, sizeof(int));
    #      // classes::cuda_add::cuda_add<<<1,1>>>( 2, 7, dev_c );
    #      CUDA_ADD.cuda_add<<<1,1>>>( 2, 7, dev_c );
    #      cudaMemcpy( &c, dev_c, sizeof(int),cudaMemcpyDeviceToHost);
    #      tv::ssprint( "2 + 7 = ",c );
    #      cudaFree( dev_c );
    #      return 0; 

    #     """).ret("int")
    #     return code


def gen_add_kernel():
    return kernel.AddKernel()



if __name__ == "__main__":
    
    test = cuda_add_wrapper()

    print(test.add_kernel.add_kernel.cuda_add())

    lib = builder.build_pybind(
        [test],
        Path(__file__).parent / "mylib")
    
    # lib.classes.cuda_add_wrapper.cuda_add_run()

    print("Code Generated!")
    # print('lib=',lib)



