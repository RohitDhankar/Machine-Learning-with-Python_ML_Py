- Source - LinkedLists - RealPython - https://realpython.com/linked-lists-python/


> Linked lists differ from lists in the way that they store elements in memory. While lists use a ```contiguous memory block``` to store references to their data, linked lists ```store references as part of their own elements.```

#
> A linked list is a collection of nodes. The ```first node``` is called the ```head```, and it’s used as the starting point for any iteration through the list. The ```last node``` must have its next reference pointing to ```None``` to determine the end of the list. 

#

> Practical Applications - Linked lists serve a variety of purposes in the real world. They can be used to implement (spoiler alert!) ```queues or stacks as well as graphs.```   
They’re also useful for much more complex tasks, such as lifecycle management for an operating system application.

- stack == Last-In/Fist-Out (LIFO)
- queue == First-In/First-Out (FIFO)
- DAG == directed acyclic graph (DAG)
- Adjacency List == list of linked lists , where each vertex of the graph is stored alongside a collection of connected vertices. 

```python

graph_dict_adj_list = {
...     1: [2, 3, None],
...     2: [4, None],
...     3: [None],
...     4: [5, 6, None],
...     5: [6, None],
...     6: [None]
... }

```

#

