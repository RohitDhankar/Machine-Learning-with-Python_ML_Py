from testPackage import *

def func1():
    from testPackage import testModule_2
    print(testModule_2.ls_3) #[1, 2, 3]

func1()    

