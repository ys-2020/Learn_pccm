from pccm import builder, core
import pickle 
from pccm.middlewares import pybind
from pathlib import Path
import pccm

from classes.cuda_add import cuda_add_class
from cumm.common import TensorView



class cuda_add_wrapper(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(cuda_add_class)
        self.add_dependency(TensorView)

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def cuda_add_run(self):
        code = pccm.FunctionCode("")
        code.raw("""
         //auto CUDA_ADD = cuda_add_class();

         int c;
         int *dev_c;
         cudaMalloc( (void**)&dev_c, sizeof(int));
         classes::cuda_add::cuda_add<<<1,1>>>( 2, 7, dev_c );
         // CUDA_ADD.cuda_add<<<1,1>>>( 2, 7, dev_c );
         cudaMemcpy( &c, dev_c, sizeof(int),cudaMemcpyDeviceToHost);
         tv::ssprint( "2 + 7 = ",c );
         cudaFree( dev_c );
         return ; 

        """).ret("void")
        return code



if __name__ == "__main__":
    
    test = cuda_add_wrapper()

    lib = builder.build_pybind(
        [test],
        Path(__file__).parent / "mylib")
    
    # lib.classes.cuda_add_wrapper.cuda_add_run()

    print("Code Generated!")



