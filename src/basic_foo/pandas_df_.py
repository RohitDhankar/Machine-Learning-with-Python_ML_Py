import pandas as pd

# df1 = pd.read_csv("./data_dir/survey_results_public.csv")
# print(df1.info(verbose=True))
# print(df1.head(3))
# print(df1.tail(3))

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
