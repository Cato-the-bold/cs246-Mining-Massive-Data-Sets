import re
import sys
import numpy as np

data_file = 'hw2-recommendations/ratings.train.txt'
loss_file = 'hw2-recommendations/result.txt'

k = 20
N_ITER = 40
LAMBDA = 0.1
learning_rate = 0.03

def error(ratings, P, Q):
    error = sum((r - np.dot(P[u],Q[i]))**2 for u,i,r in ratings)
    error += LAMBDA* np.sum(np.square(P))
    error += LAMBDA* np.sum(np.square(Q))
    return error

zero = [0]*k
def train(rating, P, Q):
    u,i,r = rating
    diff = r - np.dot(P[u], Q[i])
    d_u = 2*(-diff*Q[i]+LAMBDA*P[u])
    d_i = 2*(-diff*P[u]+LAMBDA*Q[i])

    P[u]-= learning_rate*d_u
    Q[i]-= learning_rate*d_i

    # if np.allclose(d_u, zero) and np.allclose(d_i, zero): return True
    # return False

def construct():
    ratings = np.genfromtxt(data_file, dtype=[('user','<i8'),('item','<i8'),('rating','<i8')])
    n_user = np.max(ratings['user'], axis=0)
    n_item = np.max(ratings['item'], axis=0)

    P = np.random.uniform(0, np.sqrt(5.0/k), size=(n_user+1,k))
    Q = np.random.uniform(0, np.sqrt(5.0/k), size=(n_item+1,k))

    for epoch in range(N_ITER):
        print("iter {}: {}".format(epoch, error(ratings, P, Q)))
        for rating in ratings:
            train(rating, P, Q)

construct()