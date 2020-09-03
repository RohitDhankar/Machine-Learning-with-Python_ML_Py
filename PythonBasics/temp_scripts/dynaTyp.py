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
# thus also shows that Python is not very highly - Strictly Typed . 
# CANT do this in  C++ 






# As an aside - OPERATOR OVERLOADING , is defined as - having an operator perform multiple roles as and when the 
# contextual requirement changes 
# The (+) PLUS OPERATOR - is used to Add Numbers , Concat Strings and also Merge Lists

concatStr = "str ..1 " + "str ..2 "
print(concatStr)
ls1 = ["ele1 " , "ele12 "]
ls2 = ["ele3 " , "ele14 "]
print(ls1+ls2)
# 





