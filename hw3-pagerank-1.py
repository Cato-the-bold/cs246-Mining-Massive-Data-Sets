import re
import sys
import collections
import numpy as np
from pyspark import SparkConf, SparkContext, AccumulatorParam

conf = SparkConf().setMaster("local[*]").setAppName("HW3-1").set("spark.hadoop.validateOutputSpecs", "false")
sc = SparkContext(conf=conf)

beta = 0.8
graph_file = 'graph-small.txt'
n_node = 100

MAX_ITER = 40

def define_graph(divide=True):
    data = open(graph_file)
    M = np.zeros([n_node, n_node])
    for line in data.readlines():
        s,e = line.strip().split()
        M[int(e)-1,int(s)-1] = 1.0

    if divide:
        degree = np.sum(M, axis=0, keepdims=True)
        M = M/degree

    return M

def page_rank():

    M = define_graph()
    M = sc.broadcast(M)
    R = [(i, 1.0 / n_node) for i in range(n_node)]
    R = sc.parallelize(R)

    def pagerank(kv):
        k,v = kv
        rank = [[i, v*(1-beta)/n_node] for i in range(n_node)]
        for i in range(n_node):
            rank[i][1]+= beta*M.value[i,k]*v
        return rank

    for i in range(MAX_ITER):
        msg = R.flatMap(pagerank)
        R = msg.reduceByKey(lambda x,y:x+y)

    R1 = R.map(lambda kv:(kv[1],kv[0]))
    R1 = R1.sortByKey(ascending=False).map(lambda kv: "({},{})".format(kv[1]+1,kv[0]))
    R1.coalesce(1).saveAsTextFile("_pagerank.txt")


def HITS():

    LT = define_graph(False)
    H = [(i, 1.0) for i in range(n_node)]
    H = sc.parallelize(H)

    def hits_a(kv):
        k, v = kv
        rank = [(i, LT[i,k]*v) for i in range(n_node) if LT[i,k]]
        return rank

    def hits_h(kv):
        k, v = kv
        rank = [(i, LT[k,i]*v) for i in range(n_node) if LT[k,i]]
        return rank

    for i in range(MAX_ITER):
        A = H.flatMap(hits_a).reduceByKey(lambda x, y: x + y)
        MAX_A = A.takeOrdered(1, key=lambda kv: -kv[1])
        A = A.map(lambda kv: (kv[0], kv[1]/MAX_A[0][1] ))

        H = A.flatMap(hits_h).reduceByKey(lambda x, y: x + y)
        MAX_H = H.takeOrdered(1, key=lambda kv: -kv[1])
        H = H.map(lambda kv: (kv[0], kv[1]/MAX_H[0][1] ))

    H1 = H.map(lambda kv: (kv[1], kv[0]))
    H1 = H1.sortByKey(ascending=False).map(lambda kv: "({},{})".format(kv[1] + 1, kv[0]))
    H1.coalesce(1).saveAsTextFile("hubby.txt")

    A1 = A.map(lambda kv: (kv[1], kv[0]))
    A1 = A1.sortByKey(ascending=False).map(lambda kv: "({},{})".format(kv[1] + 1, kv[0]))
    A1.coalesce(1).saveAsTextFile("authority.txt")

# page_rank()
HITS()
sc.stop()