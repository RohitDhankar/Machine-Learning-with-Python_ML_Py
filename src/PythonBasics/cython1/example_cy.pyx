# after changing code in this file - need to RE-RUN Setup.py 
cpdef int testCyThon(int someArg):
    cdef int a = 0
    cdef int k
    for k in range(someArg):
        a+= k
    return a

#testCyThon()