#### Python ( CPython) Memory Management 
#
- Source - various sources 
- https://docs.python.org/3/c-api/memory.html
- https://docs.python.org/3/reference/index.html

#

> integer objects are managed differently within the heap than strings,   
tuples or dictionaries because integers imply different storage requirements   
and speed/space tradeoffs.    
The Python memory manager thus delegates some of the work to the    
object-specific allocators, but ensures that the latter operate within the bounds of the private heap.

#
> 