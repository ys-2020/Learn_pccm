from pccm import builder, core
import pickle 
from pccm.middlewares import pybind
from pathlib import Path

from classes.init import random_init_class
from classes.check import check_class
from classes.mem_invalid import mem_class

if __name__ == "__main__":
    
    tmp1 = random_init_class()
    tmp2 = check_class()
    tmp3 = mem_class()
    lib = builder.build_pybind(
        [tmp1,tmp2,tmp3],
        Path(__file__).parent / "mylib")

    print("Code Generated!")

    




