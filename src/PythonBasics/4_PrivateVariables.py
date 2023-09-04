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


"""
#https://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes

I realize this is pretty late to the party but this link shows up on google when googling the issue. This doesn't tell the whole story. __x as a variable inside class A is actually rewritten by the compiler to _A__x, it's still not fully private and can still be accessed
"""


"""
#https://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes

As correctly mentioned by many of the comments above, let's not forget the main goal of Access Modifiers: To help users of code understand what is supposed to change and what is supposed not to. When you see a private field you don't mess around with it. So it's mostly syntactic sugar which is easily achieved in Python by the _ and __.
"""

"""
#https://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes

If you want to emulate private variables for some reason, you can always use the __ prefix from PEP 8. Python mangles the names of variables like __foo so that they're not easily visible to code outside the class that contains them (although you can get around it if you're determined enough, just like you can get around Java's protections if you work at it).

By the same convention, the _ prefix means stay away even if you're not technically prevented from doing so. You don't play around with another class's variables that look like __foo or _bar.
"""
