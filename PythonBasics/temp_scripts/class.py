import sys

class cls1():
    pass

cls1_inst1 = cls1()    
cls1_inst2 = cls1() 

cls1_inst1_RefCount = sys.getrefcount(cls1_inst1)
print(cls1_inst1) #<__main__.cls1 object at 0x7f36087f8710>
print(cls1_inst1_RefCount) #2
cls1_inst2_RefCount = sys.getrefcount(cls1_inst2)
print(cls1_inst2) #<__main__.cls1 object at 0x7f36087f8750>
print(cls1_inst2_RefCount) #2

#Instance variables --- Variables that are Specific to an INSTANCE of the Class



