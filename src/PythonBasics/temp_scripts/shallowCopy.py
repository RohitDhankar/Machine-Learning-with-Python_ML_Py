# Source - https://realpython.com/copying-python-objects/

"""
Essentially, you’ll sometimes want copies that you can modify without automatically
 modifying the original at the same time.

Assignment statements in Python do not copy objects, they create bindings 
between a target and an object. 

For collections that are mutable or 
contain mutable items, a copy is sometimes needed so one can change 
one copy without changing the other. 

This module provides generic 
shallow and deep copy operations (explained below).
"""


### SHALLOW COPY 
# Below is a vanilla Shallow , also - the copy.copy() function creates shallow copies of objects.

ls_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
ls_2 = list(ls_1) #independent from the original object
print(ls_2)
print("  "*10)
ls_1.append(["testAppend"])
print("----ls_1 == ",ls_1) 
print("----ls_2 == ",ls_2) 
#Shallow Copy independent from Original object for APPENDS
print("----ls_2 Above ---  "*3)
#
ls_1[1][1] = "RANDOM_STRING"
print("----ls_1 with RAND STR== ",ls_1) 
print("----ls_2 with RAND STR== ",ls_2) 
#Shallow Copy NOT independent from Original object for modifications within Old Child Objects
print("----ls_2 Above ---  "*3)




import copy

xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
### DEEP COPY  
## original and copy, are FULLY independent
zs = copy.deepcopy(xs)
print(xs)
print("----  "*10)
print(zs)
xs.append(["testAppend-A"])
print(" "*10)
print(xs)
print(zs)
print("----zs Above ---  "*3)
#modification to child objects in Original object - won’t affect the deep copy
