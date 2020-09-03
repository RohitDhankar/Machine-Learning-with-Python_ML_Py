"""
To understand how we can speed up Python code - we need to understand the speed-up advantages we get from STATIC TYPING
In a language which is STATIC TYPED in-place of DYANMIC TYPED - at runtime there need not be TYPE CHECKS and thus the code is excuted faster

CyThon - https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html

"""


# Python is Strongly Typed and also Dynamically Typed

# Whats a STRONGLY Typed Language - 
# Strongly typed will mean that Values stored at a location and refered by VARIABLES dont change their TYPES till 
# specifically or explicitly made to do so . 

# Examples - 
# Strongly Typed == C++ and Python
# Weakly Typed == PERL

a = 22.33 
print(type(a)) # <class 'float'>
b = 44 
print(type(b)) # <class 'int'>
c = a + b 
print(type(c)) # <class 'float'>

# As seen above we can ADD an INT to a FLOAT this is an example of OPERATOR OVERLOADING - 
# OPERATOR OVERLOADING is DIFFERENT from -  implicit conversion - of the TYPE
# CANT do this in  C++ 
# Also in Python cant add a STR to a FLOAT 

#strFloat = "str" + a #TypeError: can only concatenate str (not "float") to str
#

# Source_SO - https://stackoverflow.com/a/11328980/4928635

def to_number(x):
    """Try to convert function argument to float-type object."""
    try: 
        return float(x) 
    except (TypeError, ValueError): 
        return 0 

class Foo:
    def __init__(self, number): 
        print("--number--from__init__-",number)
        self.number = number
        print("--self.number--from__init__-",self.number)

    def __add__(self, other):
        print("----other----",other)
        print("--self.number--from__add__-",self.number)
        return self.number + to_number(other)

obj_ClsFoo = Foo(42)
#print(obj_ClsFoo) # An Object of the Class Foo == <__main__.Foo object at 0x7fd18ef24a10>

someStr = obj_ClsFoo + "str1"
print(someStr) # 42
#
str_to_num = obj_ClsFoo + "2"
print(str_to_num) # 44.0
#



# As an aside - OPERATOR OVERLOADING , is defined as - having an operator perform multiple roles as and when the 
# contextual requirement changes 
# The (+) PLUS OPERATOR - is used to Add Numbers , Concat Strings and also Merge Lists

concatStr = "str ..1 " + "str ..2 "
print(concatStr)
ls1 = ["ele1 " , "ele2 "]
ls2 = ["ele3 " , "ele4 "]
print(ls1+ls2)
# 
"""
To understand how we can speed up Python code - we need to understand the speed-up advantages we get from STATIC TYPING
In a language which is STATIC TYPED in-place of DYANMIC TYPED - at runtime there need not be TYPE CHECKS and thus the code is excuted faster

CyThon - https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html

"""





