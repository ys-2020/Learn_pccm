import pccm
from classes.cuda_add import cuda_add_class

class AddKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(cuda_add_class)
        self.add_kernel = cuda_add_class()