'''
Created on May 29, 2014

Unit tests for BST search implementation in 'solutionA' module

@author: Eric Czech
'''
import unittest

import solutionA
from solutionA import Node

class Test(unittest.TestCase):


    def validate(self, root, n, expected):
        
        # Search for the nth value in the tree
        result = solutionA.findNthGreatestValueInBST(root, n)
        
        # Verify that the result is as expected
        self.assertEqual(result, expected)
        
        # Print the results of the case validation
        print 'Problem 3 - Solution A: N = {}, Result = {}, Expected = {}'\
            .format(n, result, expected)
    

    def get_balanced_tree(self):
        """ Returns a perfectly balanced, tree of depth 4 """
        
        # Leaf nodes
        n1 = Node(None, None, 1)
        n3 = Node(None, None, 3)
        n5 = Node(None, None, 5)
        n7 = Node(None, None, 7)
        n9 = Node(None, None, 9)
        n11 = Node(None, None, 11)
        n13 = Node(None, None, 13)
        n15 = Node(None, None, 15)
        
        # 2nd level nodes
        n2 = Node(n1, n3, 2)
        n6 = Node(n5, n7, 6)
        n10 = Node(n9, n11, 10)
        n14 = Node(n13, n15, 14)
                
        # 3rd level nodes
        n4 = Node(n2, n6, 4)
        n12 = Node(n10, n14, 12)
        
        # Return root node
        return Node(n4, n12, 8)
        
    def get_problem_tree(self):
        """ Returns the example tree shown in the problem definition """
        n1 = Node(None, None, 1)
        n2 = Node(n1, None, 2)
        n4 = Node(None, None, 4)
        return Node(n2, n4, 3)
    
    def get_skewed_tree(self):
        """ Returns a similar tree to the balanced example, but with extra nodes on the right side """
        
        # Leaf nodes
        n1 = Node(None, None, 1)
        n3 = Node(None, None, 3)
        n5 = Node(None, None, 5)
        n7 = Node(None, None, 7)
        n9 = Node(None, None, 9)
        n11 = Node(None, None, 11)
        n13 = Node(None, None, 13)
        
        # Add extra nodes to right side
        n16 = Node(None, None, 16)
        n17 = Node(n16, None, 17)   
        n15 = Node(None, n17, 15)
        
        # 2nd level nodes
        n2 = Node(n1, n3, 2)
        n6 = Node(n5, n7, 6)
        n10 = Node(n9, n11, 10)
        n14 = Node(n13, n15, 14)
                
        # 3rd level nodes
        n4 = Node(n2, n6, 4)
        n12 = Node(n10, n14, 12)
        
        # Return root node
        return Node(n4, n12, 8)


    def run_tests(self, root, tree_size):
        # Test every possible, *valid* search index (1 to tree_size) 
        for i in range(1, tree_size+1):
            self.validate(root, i, i)
        
        # Test indexes outside the above range
        self.validate(root, -1, None)
        self.validate(root, 0, None)
        self.validate(root, tree_size + 1, None)
        
        # Test invalid inputs
        self.validate(None, 1, None)
        self.validate(root, None, None)
        self.validate(None, None, None)
        

    def test_solution_A_balanced_tree(self):   
        """ Tests a nice, balanced tree """     
        root = self.get_balanced_tree()
        self.run_tests(root, 15)
        
    def test_solution_A_skewed_tree(self):
        """ Tests the example tree given in the problem set """        
        root = self.get_skewed_tree()
        self.run_tests(root, 17)
        
    def test_solution_A_problem_tree(self):
        """ Tests the example tree given in the problem set """        
        root = self.get_problem_tree()
        self.run_tests(root, 4)

       
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()