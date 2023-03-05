# PageRank

Personalized PageRank is different from ordinary PageRank in a few respects:

There is the notion of a source nodes, which is what we're computing the personalization with respect to.

When initializing PageRank, instead of a uniform distribution across all nodes, the m source nodes get a mass of 1/m and every other node gets a mass of zero.

Whenever the model makes a random jump (or jumps out of a dangling node), the random jump is always back to one of the source nodes randomly (i.e., 1/m probablity to one of the source nodes); this is unlike in ordinary PageRank, where there is an equal probability of jumping to any node.

# Reference:
https://git.uwaterloo.ca/cs451/bespin/tree/master/src/main/java/io/bespin/java/mapreduce/pagerank

# Sample Output:
0.13300 367

0.12234 145

0.12140 249

0.02720 1317

0.01747 264

0.01628 266

0.01615 559

0.01610 5

0.01579 7

0.01507 251
