---
layout: single
title:  "Boyer-Moore 알고리즘(n-length array에서 majority O(n)으로 찾기)"
categories: algorithm
tag: [leetcode, algorithm]
toc: true
---

## Majority Element
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

### Example 1:
> Input: nums = [3, 2, 3]   
> Output: 3

### Example 2:
> Input: nums = [2, 2, 1, 1, 1, 2, 2]   
> Output: 2

## Moore's voting algorithm
1. This algorithm states that if any element is to be occuring more than n/2 times strictly, then I can help u
2. Steps are quite simple :
3. Initialize the possible predicted candidate for the answer to be 0, and mark its freq as 0;
4. Now iterating over the nums array, if the element==candidate, then increase its count
5. If element is different than candidate, do freq--
6. And if freq becomes 0, then update candidate, So the candidate is updated and the older one hods no longer good choice for answer.
WHY??
Because if it was a majority element then it would have not been cancelled by the other minor elements ie. As we are doing freq-- when we dont get the same element
7. The element stored in candidate is indeed our required answer!!
8. (As it is guarrenteed that answer exists, so there is no need for further checking).
9. Return ans

: 똑똑하다. 어떻게 이런 생각을 했을까. Sorting 하는 등 다른 방법들 있겠으나, O(n)으로 처리하려면 어차피 majority는 n/2보다 많으니까 count 해나가면서 같으면 +1, 다르면 -1, 0되면 숫자 바꾸고... 이런식으로 마지막에 남는게 majority라는 것.

## Reference
[leetcode - Majority Element](https://leetcode.com/problems/majority-element/description/?envType=study-plan-v2&envId=top-interview-150)