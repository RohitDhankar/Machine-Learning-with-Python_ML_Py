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
    
"""
Solutions from the net are below 
"""

"""
#https://github.com/khuongtran19/coding-challenge/tree/0cac401d49cc3eebfa142763fea8fa2d2dedbb53/leetcode-challenge/1.Two_Sum/Py
def twoSum(nums, target):
    h = {}
    for i, j in enumerate(nums):
        n = target - j
        if n not in h:
            h[j] = i
        else:
            return [h[n], i]
"""
"""
#https://github.com/jiatianzhi/LeetCode_Practice/blob/7fc8513582657f614a9069400379f6d8e4aee443/01/leetcode01.py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, m in enumerate(nums):
            n = target - m
            if n in d.values():
                return [list(d.keys())[list(d.values()).index(n)], i]
            else:
                d[i] = m


nums = [3, 3, 4, 4, 5]
target = 9
c = Solution()
x = c.twoSum(nums, target)
print (x)
"""

"""
Many Others -@20K results == https://github.com/search?q=def+twoSum%28self%2C+nums%2C+target%29%3A&type=code

- https://github.com/potatoHVAC/leetcode_challenges/blob/963dc3e290748803b8380bd2ded16e0438c11912/algorithm/1.1_two_sum.py
"""