import re
import sys
import collections

def build_frequent_set():
    rules = []
    threshold = 100

    with open('browsing.txt') as f:
        set1 = collections.defaultdict(set)
        for i, line in enumerate(f):
            for item in line.split():
                set1[item].add(i)

        set1 = {k:v for k,v in set1.items() if len(v)>=threshold}
        set2 = collections.defaultdict(set)
        for i in set1:
            for j in set1:
                if i<j:
                    s = set2[(i,j)] = set1[i] & set1[j]
                    rules.append((i,j, float(len(s))/len(set1[i])))
                    rules.append((j,i, float(len(s))/len(set1[j])))

        rules = sorted(rules, key=lambda k: -k[2])
        print(rules[:5])
        set2 = {k:v for k,v in set2.items() if len(v)>=100}


        rules = []
        set3 = collections.defaultdict(set)
        for i,j in set2:
            for p,q in set2:
                t = tuple(set(sorted([i, j, p, q])))
                if len(t)==3:
                    if t not in set3:
                        set3[t] = set2[(i,j)] & set2[(p,q)]

        for i,j,k in set3:
            s3 = set3[(i,j,k)]
            if len(s3) >= threshold:
                if (i, j) in set2: rules.append(((i, j), k, float(len(s3)) / len(set2[(i, j)])))
                if (i, k) in set2: rules.append(((i, k), j, float(len(s3)) / len(set2[(i, k)])))
                if (j, k) in set2: rules.append(((j, k), i, float(len(s3)) / len(set2[(j, k)])))

        rules = sorted(rules, key=lambda k: -k[2])
        print(rules[:5])


build_frequent_set()