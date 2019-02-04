import re
import sys
from pyspark import SparkConf, SparkContext, AccumulatorParam

conf = SparkConf().setMaster("local[*]").setAppName("HW2-2")
sc = SparkContext(conf=conf)

N = 10
centroids1 = 'hw2-q2-kmeans/c1.txt'
centroids2 = 'hw2-q2-kmeans/c2.txt'
data_file = 'hw2-q2-kmeans/data.txt'
loss_file = 'hw2-q2-kmeans/loss.txt'

MAX_ITER = 20

def l2_distance(a,b):
    return sum( (a[i]-b[i])**2 for i in range(len(a)) )

def l1_distance(a,b):
    return sum( abs(a[i]-b[i]) for i in range(len(a)) )

def assign_cluster(point, centroids, distance_func):
    Min = float('inf')
    idx = -1
    for i in range(len(centroids)):
        d = distance_func(point, centroids[i])
        if d<Min:
            Min = d; idx = i
    return idx, (Min, point)

def k_mean_train(centroids_file, distance_func):
    from shutil import copyfile
    copyfile(centroids_file, centroids_file+"0")

    def parse_line(line):
        line = line.strip().split()
        return map(float, line)

    def _sum(a, b):
        return [a[i]+b[i] for i in range(len(a))]

    def _div(a, b):
        return [a[i]/b for i in range(len(a))]

    # class StringAccumulatorParam(AccumulatorParam):
    #     def zero(self, initialValue=""):
    #         return initialValue
    #
    #     def addInPlace(self, s1, s2):
    #         return s1 + ", " + s2
    # loss_accum = sc.accumulator("", StringAccumulatorParam)

    data = sc.textFile(data_file).map(parse_line)

    for i in range(MAX_ITER):
        centroids = sc.textFile(centroids_file+str(i)).map(parse_line).collect()
        points = data.map(lambda p: assign_cluster(p, centroids, distance_func))


        clustering = points.combineByKey(lambda x:(x[0], x[1], 1),
                                             lambda x,y: (x[0]+y[0], _sum(x[1], y[1]), x[2]+1),
                                             lambda x,y: (x[0]+y[0], _sum(x[1], y[1]), x[2]+y[2])
                                             )
        new_centroids = clustering.map(lambda kv: _div(kv[1][1],kv[1][2]))
        new_centroids = new_centroids.map(lambda kv: " ".join([str(p) for p in kv]))
        new_centroids.coalesce(1, shuffle=True).saveAsTextFile(centroids_file+str(i+1))

        loss = clustering.map(lambda kv: (i, kv[1][0])).reduceByKey(lambda x,y:x+y)
        loss.coalesce(1, shuffle=True).saveAsTextFile(loss_file+str(i))
        # loss_accum.add(str(loss))


    sc.stop()

# k_mean_train(centroids1, l2_distance)
k_mean_train(centroids2, l2_distance)

# k_mean_train(centroids1, l1_distance)
# k_mean_train(centroids2, l1_distance)
