# Python does not have the Concept of Private variables as defined in PEP 8 
# there are many instances of variables declared as PRIVATE by using a PREFIX Underscore 

class A:
    def __init__(self):
        self.__var = 123
        self.__var1 = 123001
        self.__var2 = 123002
        self.__var3 = 123003

    def printVar(self):
        print(self.__var)
        print(self.__var1)
        print(self.__var3)

# create an instance of class A 

xObj = A()
print(type(xObj))
xObj.printVar()

# below fails as the PRIVATE VARIABLE == __var1 is Not Known , 
# when called from OutSide the class A


print(xObj.__var1)
"""
Traceback (most recent call last):
  File "4_PrivateVariables.py", line 24, in <module>
    print(xObj.__var1)
AttributeError: 'A' object has no attribute '__var1'
"""

"""
https://docs.python.org/3/tutorial/classes.html
Python classes provide all the standard features of Object Oriented Programming: 
#
the class inheritance mechanism allows multiple base classes, 
#
a derived class can override any methods of its base class or classes,
#
 and a method can call the method of a base class with the same name
"""

