# Source == https://docs.python.org/3.8/glossary.html

"""
DICTIONARY == An associative array, where arbitrary keys are mapped to values.
 The keys can be any object with __hash__() and __eq__() methods.
  Called a hash in Perl.
"""

# Source == https://stackoverflow.com/questions/14535730/what-does-hashable-mean-in-python

"""
An object is hashable if it has a hash value which never changes during its lifetime 
(it needs a __hash__() method), 
And can be compared to other objects (it needs an __eq__() or __cmp__() method).
 Hashable objects which compare equal must have the same hash value.
"""

# Source == https://docs.python.org/3.8/reference/datamodel.html#object.__hash__

"""
 object.__hash__(self)

    Called by built-in function hash() and for operations on members of hashed 
    collections including set, frozenset, and dict. __hash__() should return an integer.
     The only required property is that objects which compare equal have 
     the same hash value; it is advised to mix together the hash values of
      the components of the object that also play a part in comparison of 
      objects by packing them into a tuple and hashing the tuple.
"""

urTuple = (1,2,(44,55))
print(hash(urTuple)) #-2674867466080592608

# Below a LIST within a TUPLE is Not Hashable 

urTupleList = (1,2,[44,55])
#print(hash(urTupleList)) #Uncomment
"""
Traceback (most recent call last):
  File "6_Hashable_dicts.py", line 37, in <module>
    print(hash(urTupleList)) #
TypeError: unhashable type: 'list'
"""


# Below a DICT within a TUPLE is Not Hashable 

urTupleDict = (1,2,{"A":44,"B":55})
#print(hash(urTupleDict)) #Uncomment

"""
Traceback (most recent call last):
  File "6_Hashable_dicts.py", line 47, in <module>
    print(hash(urTupleDict)) #
TypeError: unhashable type: 'dict'
"""


