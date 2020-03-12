import numpy as np
d = 0.1
eps = 0.0001
A = np.array([[0, 0, 1/3, 0, 0, 0], 
         [1/2, 0, 0, 0, 0, 0], 
         [1/2, 1, 0, 0, 0, 0], 
         [0, 0, 1/3, 0, 0, 0],
         [0, 0, 1/3, 0, 0, 1],
         [0, 0, 0, 0, 1, 0]])
def calc_pagerank(d, eps, A):
    n = A.shape[0]
    A1 = np.zeros(n**2).reshape(n, n)
    ones = np.ones(n**2).reshape(n, n)
    
    for i in range(n):
        if list(A[:, i]) == list(np.zeros(6)):
            A1[:, i] = list(1/6 * np.ones(6))
        else:
            A1[:, i] = A[:, i]
        
    A2 = (1-d) * A1 + d/n * ones 
    
    r0 = np.random.rand(6, 1)
    r0 /= np.linalg.norm(r0, ord=1)
    
    prev = r0
    while True:
        r = np.dot(A2, prev)
        if np.linalg.norm(r - prev, ord=1) < eps:
            return r
        prev = r
        
print(calc_pagerank(d, eps, A))
