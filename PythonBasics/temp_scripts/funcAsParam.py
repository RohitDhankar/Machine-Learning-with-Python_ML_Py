# TODO -- terminal log files 
# Function as a Parameter for Another Func
glob = "Global variable"
def innerFunc():
    return "inner func str"

def outerFunc():
    print(innerFunc) #<function innerFunc at 0x7ff190edd680>
    print(innerFunc()) #inner func str
    print(locals()) # Initially Empty DICT = {}
    #print(globals()) # Default Global Var's 
    print(globals()["glob"])#Global variable
    return "outer func str"

def locGlob():
    """
    if no same NAME var in - Local Name Space - then execution will move out of this method
    and will search for that NAME in the Global nameSpace
    """ 
    #glob = "From Internal Local Name Space" # Toggle Comment
    print(glob)

    return   

#outerFunc()  #Uncomment
#locGlob() #Uncomment

"""
{'__name__': '__main__', 
'__doc__': None, 
'__package__': None, 
'__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7f1a589c0130>, 
'__spec__': None, 
'__annotations__': {}, 
'__builtins__': <module 'builtins' (built-in)>, 
'__file__': 'funcAsParam.py', 
'__cached__': None, 
'glob': 'Gloab vr', 
'innerFunc': <function innerFunc at 0x7f1a589f81f0>, 
'outerFunc': <function outerFunc at 0x7f1a575fa670>}

"""


