import collections


class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        #遍历记录需要变0的行和列记录

    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #计算乘积前缀和 乘积后缀和
        length=len(nums)
        pre=[1]*length
        pos=[1]*length
        res=[0]*length
        pre_sum=1
        pos_sum=1
        for i in range(length):
            pre_sum*=nums[i]
            pre[i]=pre_sum
            pos_sum*=nums[length-i-1]
            pos[length-i-1]=pos_sum
        for i in range(length):
            if i-1>=0:
                res[i]*=pre[i-1]
            if i + 1 < length:
                res[i]*=pos[i+1]
        return res






    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # 翻转后k个数字 翻转前len-k个数 再整体翻转

        length=len(nums)
        if k>length:
            k=k%length
        left=0
        right=length-k-1
        def reverse(nums,left,right):
            while left < right:
                r = nums[right]
                nums[right] = nums[left]
                nums[left] = r
                left += 1
                right -= 1
        reverse(nums, left, right)
        left = length-k
        right = length - 1
        reverse(nums, left, right)
        left = 0
        right = length - 1
        reverse(nums, left, right)
        return nums




    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(lambda x:x[0])
        res=[]
        for interval in intervals:
            if len(res)==0:
                res.append(interval)
                continue
            if res[-1][1]>=interval[0]:
                res[-1][1]=interval[1]
            else:
                res.append(interval)
        return res

    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        #遍历 a 统计长度
        #遍历b 统计长度
        #先遍历较长链表 使得两个链表等长
        #同步遍历 直到找到位置

        tempA,tempB=headA,headB
        countA=0
        countB=0
        while tempA is not None:
            tempA=tempA.next
            countA+=1
        while tempB is not None:
            tempB=tempB.next
            countB+=1
        tempA, tempB = headA, headB
        while countA>countB:
            tempA=tempA.next
            countA -= 1
        while countB>countA:
            tempB=tempB.next
            countB -= 1
        while tempA!=tempB:
            tempA = tempA.next
            tempB = tempB.next
        return tempA



    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        pre=0
        max_sub_sum=nums[0]
        for num in nums:
            pre=max(pre+num,num)
            max_sub_sum=max(max_sub_sum,pre)
        return max_sub_sum



    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 前缀和 hash表
        # 统计前缀和出现次数
        # 遍历前缀和 找到差值为k
        mp={}
        mp[0]=1
        pre_sum=0
        count=0
        for num in nums:
            pre_sum+=num
            if k-pre_sum in mp:
                count+=mp[k-pre_sum]
            if pre_sum in mp:
                mp[pre_sum]+=1
            else:
                mp[pre_sum]=1

        return count


    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        #滑动窗口
        # 先记录s的字符出现次数
        cnt=26*[0]
        tmp_cnt=26*[0]
        length_s=len(s)
        length_p=len(p)
        left=0
        res=[]
        for i in range(length_p):
            cnt[ord(p[i])-ord('a')]+=1
        for i in range(length_s):
            char=ord(s[i])-ord('a')
            tmp_cnt[char]+=1
            while tmp_cnt[char] > cnt[char]:
                tmp_cnt[ord(s[left])-ord('a')]-=1
                left+=1
            # 不超过且长度相等 就是异位词
            if i-left+1==length_p:
                res.append(left)
        return res







    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        #双指针 一边遍历 一边维护set 当有重复则更新max值以及左右指针
        check=set()
        length=len(s)
        if length ==1:
            return 1
        left=0
        res = 0
        temp_max=0
        for i in range(length):
            temp_max += 1
            if s[i] in check:
                check.remove(s[left])
                left+=1
                temp_max-=1
            check.add(s[i])
            res=max(temp_max,res)
        return res






    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res=[]
        length=len(nums)
        for i in range(length):
            num=nums[i]
            if num > 0:
                return res
            if i>0 and num==nums[i-1]:
                continue
            left=i+1
            right=length-1
            while right>left:
                target=num+nums[left]+nums[right]
                if target>0:
                    right-=1
                elif target<0:
                    left+=1
                else:
                    res.append([num,nums[left],nums[right]])
                    right -= 1
                    left += 1
                    while right>left and nums[right]==nums[right+1]:
                        right-=1
                    while right > left and nums[left] == nums[left - 1]:
                        left+=1
        return res







    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        #双指针 每次向里移动较小的index
        res=0
        left=0
        right=len(height)-1
        while left<right:
            res=max(res,min(height[left],height[right])*(right-left))
            if height[left]>height[right]:
                right-=1
            else:
                left+=1
        return res


    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        #慢指针指向非0
        #快指针扫描 非0填满慢指针
        index=0
        for i,num in enumerate(nums):
            if num is not 0:
                nums[index]=num
                index+=1
        n=len(nums)
        while index <n:
            nums[index] = 0
            index+=1





    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #
        numset=set(nums)

        res=0
        for num in nums:
            count=0
            while num+1 in numset:
                count+=1
                num+=1
            res=max(res,count)
        return res


    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        # 构建hash表 key为按字母顺序排列的计数 例如 a32b12c43
        dict = {}

        for str in strs:
            count=[0]*26
            for s in str:
                count[ord(s)-ord('a')]+=1
            key=tuple(count)
            if key not in dict:
                dict[key]=[]
            dict[key].append(str)
        return list(dict.values())





    def twoSum(self, nums, target):
        dict={}
        for index,num in enumerate(nums):
            dict[num]=index
        for index,num in enumerate(nums):
            t=target-num
            if t in dict:
                t_index = dict[t]

                if t in dict and index is not t_index:
                    return [index, t_index]




        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

Solution().productExceptSelf([1,2,3,4])