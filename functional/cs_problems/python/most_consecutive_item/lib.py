'''
Created on May 29, 2014

Solution functions to the following problem:

"A) int findMostConsecutivelyRepeatingValue (int[] arr)  
A function where the input is an array of integers and the output is the value that 
appears the most times consecutively (in a row). Ex: [1, 2, 2, 5, 5, 5, 2, 2] => 5"

There are two solutions contained, one "practical" solution (i.e. one that
you'd likely use in the real world) and one "ideal" solution (i.e. one 
more complicated solution with ideal asymptotic space and time complexity).

@author: Eric Czech
'''

import sys
from itertools import groupby
from heapq import nlargest

def findMostConsecutivelyRepeatingValue_practical(arr):
    """Determines the input list element with the most consecutive occurrences
    as well as the number of times it appears consecutively.
    
    Args:
        arr: list of elements to search
    Returns:
        A tuple in the form (most_consectively_repeating_value, number_of_consecutive_appearances)
        or None if the input list is empty
    """
    if not arr: 
        return None
    grouped = [(k, sum(1 for _ in g)) for k,g in groupby(arr)]
    return nlargest(1, grouped, key=lambda x: x[1])[0]



def findMostConsecutivelyRepeatingValue_ideal(arr):
    """Determines the input list element with the most consecutive occurrences
    as well as the number of times it appears consecutively.
    
    Args:
        arr: list of elements to search
    Returns:
        A tuple in the form (most_consectively_repeating_value, number_of_consecutive_appearances)
        or None if the input list is empty
    """
    if not arr:
        return None
    
    # Implementation here will loop through the input list only once,
    # keeping track of the maximum number of consecutive occurrences
    # as well as the item associated
    
    # Max number of consecutive occurrences 
    max_count = -sys.maxint - 1
    
    # Value associated with max number of consecutive occurrences
    max_value = None
    
    # Current number of consecutive occurrences
    curr_count = 1
    
    # The previous value seen
    last_value = None
    
    for value in arr:        
        
        # If we're not on the first element, and this element
        # is equal to the previous, increment the current number
        # of consecutive occurrences; otherwise, reset counter to 1
        if last_value is not None:
            if last_value == value:
                curr_count += 1
            else:
                curr_count = 1
                
        # Set the maximum values if we're currently at the end of the
        # longest consecutive streak seen so far
        if curr_count > max_count:
            max_value, max_count = value, curr_count 
            
        last_value = value
    
    # Return the result or None (if the array was empty)
    if max_value:
        return max_value, max_count
    else:
        return None
