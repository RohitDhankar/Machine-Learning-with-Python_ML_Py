## SOURCE -- SENTDEX -- https://www.youtube.com/watch?v=mXuEoqK4bEc
# import example_cy 
# print(type(example_cy)) #<class 'module'>
# example_cy.testCyThon(5)

import timeit

cythonCode = timeit.timeit('example_cy.testCyThon(500)',setup = 'import example_cy', number = 100000)
pythonCode = timeit.timeit('example_py.testCyThon(500)',setup = 'import example_py', number = 100000)

print(cythonCode,pythonCode)
print('CyThon = cythoCode, executed {}times faster than Python Code'.format(pythonCode/cythonCode))
# mostly 6.5 Times faster
# example_cy.testCyThon(50)==  22 Times 
# example_cy.testCyThon(500) == 138.66 Times faster

