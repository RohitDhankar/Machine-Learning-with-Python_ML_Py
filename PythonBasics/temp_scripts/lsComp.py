#List Comprehensions is the process of creating a new list from an existing one.

intsLs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# We want 'n' for each 'n' in intsLs
my_list = []
for n in intsLs:
  my_list.append(n)
print(my_list)
#
my_list = [n for n in intsLs]
print(my_list)  
#
# We want 'n*n' for each 'n' in nums
# my_list = []
# for n in nums:
#   my_list.append(n*n)
# print(my_list)
# #
# my_list = []
# print(my_list)

#
animLS = ['cat', 'dog', 'rat']
def enumerateFunc(animLS):
  for ids, animName in enumerate(animLS):
    #print('#%d: %s' % (ids + 1, animName))
    print('#%d: %s' % (ids, animName)) # 0 Index 
    print("   "*90)
#
enumerateFunc(animLS)
#
ls_ints = [0, 1, 2, 3, 4]
def sqrLsComp(ls_ints):
  sqrs = [x ** 2 for x in ls_ints]
  print(sqrs)   # 
  print("   "*90)
#  
sqrLsComp(ls_ints)  
#



