# import example_cy 
# print(type(example_cy)) #<class 'module'>
# example_cy.testCyThon(5)

import timeit

cythonCode = timeit.timeit('example_cy.testCyThon(5)',setup = 'import example_cy', number = 100000)
pythonCode = timeit.timeit('example_py.testCyThon(5)',setup = 'import example_py', number = 100000)

print(cythonCode,pythonCode)
print('CyThon = cythoCode, executed {}times faster than Python Code'.format(pythonCode/cythonCode))
# mostly 6.5 Times faster

