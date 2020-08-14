
# SOURCE == https://realpython.com/python-lambda/
# Why LAMBDA ? 
# LAMBDA Functions also referred to as -  single expression functions.


def addFunc(int_a,int_b):
    return int_a + int_b

addFunc(22,33)
result = addFunc(22,33)
print("result of addFunc = ",result)

#
print("Within Script Terminal prints - lambda's wont print results without a print()")
print((lambda x, y: x + y)(22,33))

#
print("Within Script Terminal prints - lambda's wont print results without a print()")
add = lambda x, y: x + y
print(add(22,11))

"""
# Can be run dircetly at the terminal as below - 
Python 3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
>>> add = lambda x, y: x + y
>>> add(22,11)
33
>>> 
"""