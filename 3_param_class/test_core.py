import shutil
from pathlib import Path

import pccm
from pccm import builder, core
from pccm.middlewares import pybind
from classes.mod import PbTestVirtual, Test3, Test4, OtherSimpleClass
import pickle 

def test_core():
    cu = Test4()
    cu2 = OtherSimpleClass()

    cu_scratch = core.Class()
    scratch_meta = core.StaticMemberFunctionMeta(name="scratch_func")
    scratch_code_obj = core.FunctionCode("")
    scratch_code_obj.raw("""
    return 50051;
    """).ret("int")
    cu_scratch.add_func_decl(core.FunctionDecl(scratch_meta, scratch_code_obj))
    cu_scratch.namespace = "scratch"
    cu_scratch.class_name = "ScratchClass"

    # cu_scrach will be like:
    
    # # include <scratch/ScratchClass.h>
    # namespace scratch {
    # int ScratchClass::scratch_func()   {
    #   return 50051;
    # }
    # } // namespace scratch

    lib = builder.build_pybind(
        [cu_scratch,cu, cu2, PbTestVirtual()],
        Path(__file__).parent / "mylib")
    
    assert lib.classes.mod.Test4.add_static(1, 2) == 3
    assert not hasattr(lib.classes.mod.Test4, "invalid_method")
    t3 = lib.classes.mod.Test3()
    t3.square_prop = 5
    assert t3.square_prop == 25

    class VirtualClass(lib.classes.mod.PbTestVirtual):
        def func_0(self):
            self.a = 42
            return 0

        def func_2(self, a: int, b: int):
            self.a = a + b
            return 0

    vobj = VirtualClass()
    assert vobj.a == 0
    vobj.run_virtual_func_0()
    assert vobj.a == 42
    vobj.run_pure_virtual_func_2(3, 4)
    assert vobj.a == 7

    assert vobj.EnumExample.kValue1.name == "kValue1"
    assert vobj.EnumExample.kValue1.value == 1
    assert vobj.EnumExample.kValue1 | vobj.EnumExample.kValue2 == 3
    assert vobj.kValue1 | vobj.kValue2 == 3

    tsc = lib.classes.mod.OtherSimpleClass()
    tsc.a = 456
    tsc_bytes = pickle.dumps(tsc)

    tsc_recover = pickle.loads(tsc_bytes)
    assert tsc_recover.a == tsc.a

    return lib


if __name__ == "__main__":
    lib = test_core()
    
    print("Check Pass!")
