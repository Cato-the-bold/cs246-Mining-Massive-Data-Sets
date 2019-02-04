import re
import sys
import collections
import numpy as np

beta = 0.8
graph_file = 'graph-small.txt'
n_node = 100

MAX_ITER = 40

def define_graph():
    data = open(graph_file)
    M = np.zeros([n_node, n_node])
    for line in data.readlines():
        s,e = line.strip().split()
        M[int(e)-1,int(s)-1] = 1

    degree = np.sum(M, axis=0, keepdims=True)
    M = M/degree

    return M

def page_rank():

    M = define_graph()
    # M = sc.broadcast(m)
    R = {i: (1 / n_node, M[i,:]) for i in range(n_node)}
    R = sc.parallelize(R)

    def pagerank(kv):
        k,(v,m) = kv
        r = (1-beta)/n_node
        v1 = beta*np.dot(m, R)
        return (k,v1+r)

    for i in range(MAX_ITER):
        R = R.map(pagerank)

    R1 = R.map(lambda kv:(kv[1],kv[0]))
    R1 = R1.sortByKey(ascending=False)
    R1.take(5).coalesce(1).saveAsTextFile("top-5.txt")

    R1 = R1.sortByKey()
    R1.take(5).coalesce(1).saveAsTextFile("bottom-5.txt")

    sc.stop()

page_rank()
