'''
Created on May 29, 2014

Solution functions for the following problem:

"Write a function where the input is an array of integers and the output is the value that 
appears most frequently. Ex: [1, 2, 2, 5, 5, 5, 2, 2] => 2"

There are two solutions contained, one "practical" solution (i.e. one that
you'd likely use in the real world) and one "ideal" solution (i.e. one 
more complicated solution with a configurable asymptotic space complexity).

@author: Eric Czech
'''

import math
import hashlib
from collections import Counter

def findMostFrequentValue_practical(arr):
    """Determines the input list element with the most occurrences.
    
    Args:
        arr: list of elements to search
    Returns:
        A tuple in the form (most_frequent_value, frequency)
        or None if the input list is empty
    """
    result = Counter(arr).most_common(1)
    return result[0] if result else None


def findMostFrequentValue_ideal(arr, c=.5):
    """Determines the input list element with the most occurrences.
    
    This implementation is "space-optimized" to an extent determined by the setting 
    of the 'c' parameter.  This parameter controls both the computational and space 
    complexity of this routine where the computational complexity = O( n^( 1+c ) ), 
    and the storage complexity = O( n^( 1-c ) + n^c ).
    
    Here are the characteristics of some common settings for 'c' (must be between 0 and .5):
    A) 0    --> High speed - O(n), High storage overhead - O(n):
            equivalent to computing a single histogram (i.e. the naive approach)
    B) .25  --> Medium speed - O(n^1.25), Medium storage overhead - O(n^.75 + n^.25)
    B) .5   --> Low speed - O(n^1.5), Low storage overhead - O(n^.5)
    
    * The lowest possible storage overhead is O(sqrt(n)); increasing the value of 'c'
    when c is between 0 and .5 lowers the space required, but increasing 'c' beyond .5
    actually starts to increase the required space (i.e. O( n^( 1-c ) + n^c ) is minimized 
    when c = .5) while ALSO increasing the time complexity.  In other words, the space and
    time complexities are inversely correlated for values of 'c' between 0 and .5, but positively
    correlated when 'c' > .5 (so setting 'c' greater than .5 doesn't help performance in any way).
    As such, 'c' will automatically be set within the closest endpoint on the range [0, .5] if 
    it is specified outside of that range.
    
    Args:
        arr: list of elements to search
        c: storage usage parameter; must be between 0 and 1/2
    Returns:
        A tuple in the form (most_frequent_value, frequency)
        or None if the input list is empty
    """
    if not arr:
        return None
    
    # Limit c to the restricted range described above
    c = min(max(0, c), .5)
    
    # Since we're going through the trouble of implementing a solution to
    # this problem that doesn't necessarily require O(n) space, let's also assume
    # that the input would really be an iterator in a real-life scenario
    # where the number of elements contained is too large to put in memory
    # at once (i.e. as a list/array).  
    arr_iter = iter(arr)
    
    # Get the total number of elements 
    n = sum(1 for _ in arr_iter)
    
    # Compute the number of buckets necessary, as a function of c
    num_buckets = int(math.ceil(n ** c))
    
    # Create N^c buckets and associate items with each bucket using 
    # a cryptographic hash function with approximately uniformly distributed
    # results regardless of the distribution of the input elements (md5 in this case).
    #
    # This will yield approximately the same number of items, n ^ (1-c), within each bucket.
    #
    # Then, for each bucket of potential size n ^ (1-c), loop over the entire set of input elements
    # and determine the most frequent item for all items that are members of that bucket
    # (and add that result to a list of results).
        
    results = []
    for search_bucket in range(num_buckets):  
        # Reset the iterator
        arr_iter = iter(arr)
        
        frequencies = Counter()
        for item in arr_iter:
            # Get the bucket number for the item and add it to the 
            # frequency counts if its bucket matches the one we're searching for
            bucket = int(hashlib.md5(str(item)).hexdigest(), 16) % num_buckets
            if search_bucket == bucket:
                frequencies[item] += 1

        # Add the most common item to the results
        result = frequencies.most_common(1)
        if result:
            results.append(result[0]) 
    
    # Return the top result in the results across all buckets
    return sorted(results, key=lambda x: x[1], reverse=True)[0]

