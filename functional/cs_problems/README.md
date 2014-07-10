# CS Problem Solutions

_* This project was completed as part of a 24-hour coding challenge I went through recently_


The python modules in this project contain solutions to several common computer science problems.  Each of those problems is detailed below and the modules themselves are meant to address a single problem at a time.

Every module also has a companion set of unit tests to show example usage and verify that the solutions are correct.

Also, for most of the solutions I tried to include both a "practical" way to solve the problem as well as a more theoretically "ideal" (in space/time complexity) solution.  Having two solutions was also convenient for unit testing where the results of a known library could be compared against my custom functions to make sure they're semantically correct (I do this frequently in a production setting too).

\* *Apologies for any style inconsistencies or weird idioms -- I'm much more of a java person but am trying to become pythonic on a production-worthy level ASAP.*

Solutions and Explanations:
* [Solution API](#solution-api)
* [Problem 1 - Consecutive Elements](#problem-1---consecutive-elements)
* [Problem 2 - Frequent Elements](#problem-2---frequent-elements)
* [Problem 3 - BST Rank](#problem-3---binary-search-tree-rank)

***
## Solution API

As a quicker, easier way to look at some of the results produced by the code answering these questions, I put some 
of the solutions behind a REST API.  I thought this would be easier than trying to make sure reviewers have a compatible python environment with the source code in the project.

I'll leave the explanations of each solution to the sections following, but as a quick reference, here's what's immediately available from the API:

* Querying for the item in a list with the largest number of consecutive appearances
    * http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/most_consecutive/1,2,2,5,5,5,2,2
    * [http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/most_consecutive/1,2,,,1,1,1 ,1 ,asdf,3,3](http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/most_consecutive/1,2,,,1,1,1 ,1 ,asdf,3,3)
* Querying for the item in a list that occurs most frequently
    * http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/most_frequent/1,2,2,5,5,5,2,2
    * http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/most_frequent/1,2,1,2,2,3,1,1,1,1,1,asdf,3,2,3
* Merge-sorting two lists
    * http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/merge_sort/?list1=1,3,5&list2=2,4,6
    * http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/merge_sort/?list1=0,1,2,3,4&list2=1,3,6,8
    * http://ec2-50-112-200-1.us-west-2.compute.amazonaws.com:5000/merge_sort/?list1=1,kdl,3,5&list2=2,asdf,,,4,6
    
The API is deployed on a free-tier, micro AWS instance.

The source code for the API server is here: [api/server.py](python/api/server.py)

***
## Problem 1 - Consecutive Elements

> Q: Write a function where the input is an array of integers and the output is the value that 
> appears the most times consecutively (in a row). Ex: [1, 2, 2, 5, 5, 5, 2, 2] => 5
> What is the Big-O complexity of your solution (in both time and space)? Why?


Solutions in : [most_consecutive_item/lib.py](python/most_consecutive_item/lib.py)

Unit tests in: [most_consecutive_item/test.py](python/most_consecutive_item/test.py)  (Executable via a main method)


A "practical" solution to this problem, like this:

```
# arr is input element array
grouped = [(k, sum(1 for _ in g)) for k,g in groupby(arr)]
return nlargest(1, grouped, key=lambda x: x[1])[0]
```

would usually run with a computational complexity of **O(n log(n))** (due to sorting) and a storage complexity of **O(n)**.  That's not great but I'd always prefer a solution like this due to its simplicity unless there was a really good reason not to.  

Assuming there is a good reason to write a more optimal, custom solution like the one I gave, then you can shrink those complexities down to **O(n)** runtime, **O(1)** storage.

***
## Problem 2 - Frequent Elements


> Q: Write a function where the input is an array of integers and the output is the value that
> appears most frequently. Ex: [1, 2, 2, 5, 5, 5, 2, 2] => 2
> What is the Big-O complexity of your solution (in both time and space)? Why?


Solutions  ==> [most_frequent_item/lib.py](python/most_frequent_item/lib.py)

Unit tests ==> [most_frequent_item/test.py](python/most_frequent_item/test.py)  (Executable via a main method)


A "practical" solution to this problem, like this:

```
# arr is input element array
result = collections.Counter(arr).most_common(1)
return result[0] if result else None
```

would usually run with a computational complexity of **O(n)** and a storage complexity of **O(n)**.  This makes sense because the most naive solution just involves computing a histogram of element counts and then finding the value with the maximum number of occurrences in that histogram after adding everything to it (max functions run in linear time).

In extreme cases where the number of items is really big, the solution I proposed makes some tradeoffs between time and space complexity.  The solution includes a parameter, *c*, used to manipulate that tradeoff on a continuous spectrum.  I think this kind of control is incredibly important for real-world applications like this where the *real* problem isn't getting the space complexity down, it's getting things to work either completely in memory or with a combination of memory and **sequential** disk I/O.  In other words, I'll take an algorithm with **O(n^2)** space complexity over one with **O(n)** any day if **n** is to big too fit in memory anyways, and the former allows for sequential I/O in and out of memory while the latter doesn't (i.e. random I/O required).  The performance between these two I/O methods differs by approximately two orders of magnitude, so the exponents of the space complexities would have to be at least that different before sequential algorithms using more space would *actually* take longer. 

Anyways, the solution I proposed would only ever involve sequential I/O in and out of RAM and would also allow, at least to some extent, for a trade off to be made between space and time to try to fit everything in memory instead of spilling anything to disk (or a network).

The computational complexity of the solution is **O( n^( 1 + *c* ) )**, where *c* is between 0 and .5.  The storage complexity, **O( n^( 1 - *c* ) + n^*c* )**, is also a function of *c* minimized at *c* = .5.  This means that the best this solution can do to save space is to reduce the storage complexity to **O(sqrt(n))**, at the expense of an increased runtime complexity of **O( n ^ (3/2) )**.  At the opposite end of the spectrum when *c* = 0, the runtime and storage complexities are both **O(n)**, and the implementation becomes essentially identical to the naive histogram approach I mentioned above.

***
## Problem 3 - Binary Search Tree Rank

> Q: Write a function where the input is a binary search tree of integers and the
> output is the Nth-greatest value in the tree. The BST is defined by a root node 
> and the property: every node on the right subtree has to be larger than the
> current node and every node on the left subtree has to be smaller (or equal) than 
> the current node.
> What is the Big-O complexity of your solution (in both time and space)? Why?

Solutions in : [binary_search_tree_rank/lib.py](python/binary_search_tree_rank/lib.py)

Unit tests in: [binary_search_tree_rank/test.py](python/binary_search_tree_rank/test.py)  (Executable via a main method)

This solution works by moving recursively, depth-first, and left-first throughout the tree to enumerate the nodes in order from least to greatest.  Starting with the node having the smallest value, a counter is incremented within the traversal until **N** nodes are encountered, at which point the value for that node is returned.

The space and time complexity for this approach are both **O(|E| + |V|)** since the traversal involves visiting each vertex across each edge and the graph representation doesn't involve matrices or something else requiring **O(n^2)** (or **O(|V| ^ 2)**) space. 

