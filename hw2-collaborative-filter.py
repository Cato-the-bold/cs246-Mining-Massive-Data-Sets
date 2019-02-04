import re
import sys
import numpy as np

user_shows_file = 'q1-dataset/user-shows.txt'
shows_file = 'q1-dataset/shows.txt'


def user_user_filter(data, P, Q):
    sim = np.matmul(data, data.T)
    sim = sim/P
    sim = sim/P.T
    return np.matmul(sim,data)

def item_item_filter(data, P, Q):
    sim = np.matmul(data.T, data)
    sim = sim/Q
    sim = sim/Q.T
    return np.matmul(data, sim)

def construct():
    data = np.genfromtxt(user_shows_file, dtype=None)
    shows = open(shows_file).readlines()
    P = np.sum(data, axis=1, keepdims=True)
    Q = np.sum(data, axis=0, keepdims=True)
    P = np.sqrt(P)
    Q = np.sqrt(Q)

    alexa_id = 500
    R = user_user_filter(data, P, Q)
    scores = [ (R[alexa_id, i], i) for i in range(100)]
    scores = sorted(scores, key=lambda x: -x[0])
    rec = [ shows[scores[i][1]] for i in range(5)]
    print(rec)

    R = item_item_filter(data, P, Q)
    scores = [ (R[alexa_id, i], i) for i in range(data.shape[1])]
    scores = sorted(scores, key=lambda x: -x[0])
    rec = [ shows[scores[i][1]] for i in range(5)]
    print(rec)

construct()