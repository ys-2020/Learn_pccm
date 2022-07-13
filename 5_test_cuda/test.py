# —————— It is old-fasioned —————— #
# import imp
# file, pathname, desc = imp.find_module('pyifiles')
# lib = imp.load_module('pyifiles', file, pathname, desc)

# —————— New Version —————— #
import importlib

module_spec = importlib.util.find_spec('mylib')
lib = importlib.util.module_from_spec(module_spec)
print(lib)

a = 1
b = 2
c = 3
d = 4

print("a =",a)
print("b =",b)
print("c =",c)
print("d =",d)


print("a+b+c+d =",lib.classes.Test2_3_file.Test3.test3_sum_of_4(a,b,c,d))

print("(a+b)*c-d =",lib.classes.Test2_3_file.Test3.test2_cal(a,b,c,d))






