#!/usr/bin/env python3

# A DIR which has Python Scripts / Python Modules can be Converted into a Python Package 
# The advantage of creating a Package - 
# FOOBAR__Further Reading REQUIRED - https://www.python.org/dev/peps/pep-0420/


#from PythonBasics.testPackage import *
#ModuleNotFoundError: No module named 'PythonBasics'
#from ..testPackage import *
from testPackage import *

def testPackFunc():
    print(testPackage.ls_1)
    return

testPackFunc()

