###### NameSpaces and Scope 

- Source - https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces

> A ```namespace``` is a mapping from names to objects.     
 Most namespaces are currently implemented as ```Python dictionaries```, but that’s normally not noticeable in any way (except for performance),      
 and it may change in the future.   
 Examples of namespaces are:    
 the set of built-in names (containing functions such as abs(),    
 and built-in exception names);      
 the global names in a module;     
 and the local names in a function invocation.     
 In a sense the ```set of attributes of an object also form a namespace.```     
 The important thing to know about namespaces is that there is absolutely no relation    
 between names in different namespaces; for instance, ```two different modules```         
 may both define a function maximize without confusion — users of the modules must ```prefix it with the module name.```

 #
 - Source - https://docs.python.org/3/tutorial/classes.html

 > Namespaces are ```created at different moments``` and have ```different lifetimes```. The namespace containing the built-in names is created when the ```Python interpreter starts up```, and is never deleted. The ```global namespace``` for a module is created when the module definition is read in; normally, module namespaces also last until the ```interpreter quits```.

#

- Source - https://www.python-course.eu/namespaces.php

 > Generally speaking, ```a namespace (sometimes also called a context)``` is a naming system for making names unique to avoid ambiguity. Everybody knows a namespacing system from daily life, i.e. the naming of people in firstname and familiy name (surname). Another example is a ```network: each network device (workstation, server, printer, ...) needs a unique name and address.``` Yet another example is the directory structure of ```file systems```. The same file name can be used in different directories, the files can be uniquely accessed via the pathnames. 

 #

#
- Source - https://realpython.com/python-namespaces-scope/

> Enclosing NameSpace - When the main program calls f(), Python creates a new namespace for f(). Similarly, when f() calls g(), g() gets its own separate namespace. The namespace created for g() is the local namespace, and the namespace created for f() is the ```enclosing namespace.```

