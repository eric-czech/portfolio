'''
Created on May 29, 2014

Unit tests for 'most_frequent_item' library, defining methods for determining
the most frequently occurring element in a list

@author: Eric Czech
'''
import unittest

from lib import *

class Test(unittest.TestCase):


    def validate(self, arr, expected):
        
        # Get results for both solutions
        ideal_res = findMostFrequentValue_ideal(arr)
        practical_res = findMostFrequentValue_practical(arr)
        
        # Verify that the results from both methods are the list of expected results
        self.assertIn(ideal_res, expected)
        self.assertIn(practical_res, expected)
        
        # Print the results of the case validation
        print  'Most frequent item test case: input = {}, result = {} ({} appearances)'\
            .format(
                    arr, 
                    ideal_res[0] if ideal_res else None, 
                    ideal_res[1] if ideal_res else None
            )
    
    def test_most_frequent_item(self):
        # Edge cases for empty inputs
        self.validate(None, [None])
        self.validate([], [None])
        
        # Simple cases
        self.validate([1], [(1, 1)])
        self.validate([1, 2, 2, 2, 2, 1], [(2, 4)])
        self.validate([1, 2, 2, 2, 8, 2, 7, 8, 8 , 8, 8, 1], [(8, 5)])
        
        # Tougher case with multiple correct answers
        self.validate([1, 2, 2, 2, 3, 3, 3], [(3, 3), (2, 3)])
        
        # Case in problem definition
        self.validate([1, 2, 2, 5, 5, 5, 2, 2], [(2, 4)])
        
        

if __name__ == "__main__":
    unittest.main()
