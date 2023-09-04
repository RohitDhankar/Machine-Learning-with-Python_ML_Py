#TODO -- starts Corey code 
nums = [1,2,3,4,5,6,7,8,9,10]
ls_com = [n for n in nums]
print(ls_com)
ls_com_1 = [n*2 for n in nums] # Squares of All NUMS
print(ls_com_1)
ls_com_2 = [n for n in nums if n%2 ==0] # All Even NUMS Only
print(ls_com_2)
# 
"""
Get a Tuple each -- for  a Letter + Number Pair == (letter,number_int)
Letters to come from string -- abcd
Nums to come from List -- 0123

"""
print("  "*90)
ls_com_3 = [(str1,num_int) for str1 in "abcd" for num_int in range(4)] #
print(ls_com_3)
#
print("  "*90)
ls_vals = [["Nested_LS_1"],[1,2,3],["Nested_LS_2"],["Nested_LS_3"]]
ls_com_4 = [{str_keys,nested_lists} for str_keys in "abcd" for nested_lists in ls_vals] #
print(ls_com_4)

#TODO -- Corey code below ? 


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



