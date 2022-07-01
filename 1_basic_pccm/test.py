# —————— It is old-fasioned —————— #
# import imp
# file, pathname, desc = imp.find_module('pyifiles')
# lib = imp.load_module('pyifiles', file, pathname, desc)

# —————— New Version —————— #
import importlib

module_spec = importlib.util.find_spec('pyifiles')
lib = importlib.util.module_from_spec(module_spec)
print(lib)

a = 1
b = 2    

print(lib.classes.test_class.Test1.add(a, b))






