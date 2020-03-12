import numpy as np
import sys

def load_text(filename):
    data = np.loadtxt(filename, comments='#', dtype=int)
    N = np.max(data)

    ret = np.zeros((N+1, N+1), dtype=float)
    for i in range(data.shape[0]):
        ret[data[i][0]][data[i][1]] = 1

    for j in range(N+1):
        if np.sum(ret[j] != 0):
            ret[j] = ret[j]/np.sum(ret[j])
    
    return ret.T
    

def calc_pagerank(d, eps, A):
    n = A.shape[0]
    A1 = np.zeros(n**2).reshape(n, n)
    ones = np.ones(n**2).reshape(n, n)
    zeros = list(np.zeros(n))
    
    for i in range(n):
        if list(A[:, i]) == zeros:
            A1[:, i] = list(1/n * np.ones(n))
        else:
            A1[:, i] = A[:, i]
        
    A2 = (1-d) * A1 + d/n * ones 
    
    r0 = np.random.rand(n, 1)
    r0 /= np.linalg.norm(r0, ord=1)
    
    prev = r0
    while True:
        r = np.dot(A2, prev)
        if np.linalg.norm(r - prev, ord=1) < eps:
            return r
        prev = r

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(u"使い方: %s FILENAME" % sys.argv[0])
        sys.exit(1)
    d = 0.1
    eps = 0.0001
    A = load_text(sys.argv[1])
    print(calc_pagerank(d, eps, A))
