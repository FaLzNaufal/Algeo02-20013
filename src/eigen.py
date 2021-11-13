import numpy as np

def eig(a):
    pQ = np.eye(a.shape[0])
    temp = a.copy()
    for i in range(10):
        Q, R = np.linalg.qr(temp)
        pQ = pQ @ Q
        temp = R @ Q
    return np.diag(temp), pQ