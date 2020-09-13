"""
Given an array of integers nums and an integer target, 
return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, 
and you may not use the same element twice.

You can return the answer in any order.
"""
#ls_nums = [11,22,44,99,199,899,677,455,11,777] # Multiple same ele
ls_nums = [11,22,44,99,199,899,677,455,777] 
int_target = 210 # 199 ( Idx =4) + 11 ( Idx =0)
def twoSum(ls_nums,int_target):
    ls_idx = []
    cnt = 0 
    while True:
        try:
            for j , k in enumerate(ls_nums):
                if ls_nums[cnt] + int(k) == int_target:
                    #print("-got target---",int_target)
                    ls_idx.append(ls_nums[cnt])
                    ls_idx.append(int(k))
                if j == 8:
                    cnt += 1
                    if cnt == 9:
                        pass
        except:
            #print("-double entry in list-",ls_idx)
            return ls_idx[0:2]
            break

ls_result = twoSum(ls_nums,int_target)
print(ls_result)

#for i in range(-2,len_ls):
# negative start index (-2) ensures we iterate over all Elements
    