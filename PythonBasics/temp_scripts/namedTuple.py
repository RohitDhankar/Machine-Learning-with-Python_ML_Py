#collections.namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)
# https://docs.python.org/3/library/collections.html#collections.namedtuple

#Source - https://realpython.com/lessons/collectionsnamedtuple/

from collections import namedtuple

Car = namedtuple("Car", ["color", "make", "model", "mileage"])
my_car = Car(color="midnight silver", make="Tesla", model="Model Y", mileage=5)
print(my_car.color)
#'midnight silver'


