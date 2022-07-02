from pccm import builder, core
import pickle 
from pccm.middlewares import pybind
from pathlib import Path

from classes.Test1_file import Test1
from classes.Test2_3_file import Test2,Test3

if __name__ == "__main__":
    
    test3 = Test3()

    lib = builder.build_pybind(
        [test3],
        Path(__file__).parent / "mylib")

    print("Code Generated!")
    # print('lib=',lib)



