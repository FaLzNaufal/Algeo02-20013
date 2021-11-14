import numpy as np

def eigen(matriks, nIteration = 10):
    eigVec = np.eye(matriks.shape[0])
    eigVal = matriks.copy()
    for _ in range(nIteration):
        Q, R = np.linalg.qr(eigVal)
        eigVec = eigVec * Q
        eigVal = R * Q
    return np.diag(eigVal), eigVec