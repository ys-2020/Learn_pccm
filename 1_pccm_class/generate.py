from pccm import builder, core
import pickle 
from pccm.middlewares import pybind
from pathlib import Path

from classes.test_class import Test1

if __name__ == "__main__":
    
    test1 = Test1()
    print(test1.add())

    lib = builder.build_pybind(
        [test1],
        Path(__file__).parent / "mylib")



    print("Code Generated!")
    # print('lib=',lib)



