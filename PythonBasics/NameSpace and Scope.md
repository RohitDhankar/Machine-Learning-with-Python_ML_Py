###### NameSpaces and Scope 

- Source - https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces

> A ```namespace``` is a mapping from names to objects.     
 Most namespaces are currently implemented as ```Python dictionaries```, but that’s normally not noticeable in any way (except for performance),      
 and it may change in the future.     
 
  Examples of namespaces are: the set of built-in names (containing functions such as abs(), and built-in exception names); the global names in a module; and the local names in a function invocation. In a sense the set of attributes of an object also form a namespace. The important thing to know about namespaces is that there is absolutely no relation between names in different namespaces; for instance, two different modules may both define a function maximize without confusion — users of the modules must prefix it with the module name.
