# 1- Check the REF Count for a Variable
# 2- Learn when a Variable is removed from the Heap 

import sys

myVar = "someStr"
print(myVar.__init__) #<method-wrapper '__init__' of str object at 0x7f37c3f99130>
myVarRefCount = sys.getrefcount(myVar)
print(myVarRefCount) # prints - 4 

myVar1 = "someStr"
print(myVar1.__init__) #<method-wrapper '__init__' of str object at 0x7f37c3f99130>
myVarRefCount1 = sys.getrefcount(myVar1)
print(myVarRefCount1) # prints - 5

myVar2 = "someStr"
print(myVar2.__init__) #<method-wrapper '__init__' of str object at 0x7f37c3f99130>
myVarRefCount2 = sys.getrefcount(myVar2)
print(myVarRefCount2) # prints - 6

"""
1 Object in Memory == someStr #str object at 0x7f37c3f99130
3 Names = myVar , myVar1 and myVar2 used for the same Object == someStr
6 Rerences - 3 Explicit and 3 Implicit ( Which Ones ?)

"""
