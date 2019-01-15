import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("HW1")
sc = SparkContext(conf=conf)

N = 10
input_file = 'soc-LiveJournal1Adj.txt'
output_file = 'result.txt'

lines = sc.textFile(input_file)

def parse_line(line):
    line = line.strip().split()
    res = []
    if len(line)>1:
        (user, friends) = line
        friends = friends.split(",")
        for f1 in friends:
            for f2 in friends:
                if f1!=f2:
                    res.append(((f1,f2), 1))
    return res
#u1 u2,u3:  u2 and u3 have a common friend u1 => Map: (u2,u3),1  => Reduce: (u2,u3), cnt  => Map: u2, (u3, cnt)
#   => Group: u2, ((u3,c1),(u1,c2))  => Sort&Map  =>  u2, (u1,u3)
pairs = lines.flatMap(parse_line).reduceByKey(lambda x,y: x+y)
pairs2 = pairs.map(lambda kv: (kv[0][0],(kv[0][1],kv[1]))).sortByKey()
recommendation = pairs2.groupByKey().mapValues(list).map(lambda kv: (kv[0], sorted(kv[1], key=lambda v: -v[1])[:N]))
recommendation = recommendation.map(lambda kv: (kv[0], [v[0] for v in kv[1]]))
recommendation = recommendation.map(lambda kv: str(kv[0])+'\t'+",".join([str(friend) for friend in kv[1]]))
recommendation.saveAsTextFile(output_file)

sc.stop()
