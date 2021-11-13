import numpy as np
import math
from PIL import Image
from scipy.linalg.decomp_svd import null_space
from matrix import *
from numpy import linalg as LA

def readImage():
    img = Image.open('D:\ProjectKuliah\AlGeo\Algeo02-20013\src\pepe.png')
    imgMatrix = np.array(img.convert("L"))
    return imgMatrix

def transpose(matrix): #tidak dipakai, jadinya pakai fungsi bawaan numpy
    return np.array([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])

def multiply(matrix1, matrix2): #tidak dipakai, jadinya pakai fungsi bawaan numpy
    return np.array([[sum([matrix1[i][k]*matrix2[k][j] for k in range(len(matrix1[0]))]) for j in range(len(matrix2[0]))] for i in range(len(matrix1))])

def getSigma(matrixA, sigmaValues):
    x = np.array([[float(0) for j in range(len(matrixA[0]))] for i in range(len(matrixA))])
    
    for i in range(len(x)):
        if (i >= len(x[0])):
            break
        else:
            x[i][i] = sigmaValues[i]
    return x

def getSigmaValues(eigenvalues):
    sigmaValues = np.array(eigenvalues, copy=True)
    for i in range(len(eigenvalues)):
        sigmaValues[i] = math.sqrt(abs(eigenvalues[i]))
    return sigmaValues

def getUWithNullSpace(matrix, matrixAtA, eigenValues, sigmaValues): #tidak terpakai
    u = np.array([[] for i in range(len(matrix))])
    
    for i in range(len(sigmaValues)):
        tmp = np.array(matrixAtA, copy=True)
        for j in range(len(tmp)):
            tmp[j][j] -= eigenValues[i]
        ns = null_space(tmp)
        ns = ns * np.sign(ns[0,0])
        tempU = (1/sigmaValues[i])*matrix
        # print(np.matmul(tempU, ns))
        u = np.append(u, np.matmul(tempU, ns), axis = 1)
    # u = np.transpose(np.array(u))
    return(u)

def getU(matrix, eigenvector, sigmaValues):
    u = []
    eigenVectorTransposed = np.transpose(eigenvector)
    #print("eig = ",eigenVectorTransposed[0].shape)
    for i in range(len(sigmaValues)):
        tempU = (1/sigmaValues[i])*matrix
        tempUTransposed = np.transpose(tempU)
        x = np.matmul(eigenVectorTransposed[i], tempUTransposed)
        u.append(x)
    u = np.transpose(np.array(u))
    return(u)

def svd2(A): #ini gajadi
    lam, v = LA.eig(A.T @ A)

    V = v[:, lam.argsort()[::-1]]

    lam_sorted = np.sort(lam)[::-1]
    lam_sorted = lam_sorted[lam_sorted > 1e-8]
    sigma = np.sqrt(lam_sorted)
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    Sigma[:min(A.shape[0],A.shape[1]), :min(A.shape[0],A.shape[1])] = np.diag(sigma)

    # print("Sigma=", np.round(Sigma, 4))
    # print("V=", np.round(V, 4))
    r = len(sigma)
    U = A @ V[:,:r] / sigma
    # print("U=", np.round(U, 4))
    a = np.arange(12).reshape(3, 4)
    # print("a=",a)
    # print("sigma=",Sigma)
    # print("deleted = ",np.delete(Sigma, np.s_[(len(U[0])):], 0))
    Sigma = np.delete(Sigma, np.s_[(len(U[0])):], 0)
    return U, Sigma, np.transpose(V)

def svd(matrix):
    transposed = np.transpose(matrix)
    ata = np.matmul(transposed, matrix)
    eigenvalue, eigenvector = np.linalg.eig(ata)
    eigenvalue = np.real(eigenvalue)
    eigenvector =  np.real(eigenvector)
    eigenvalue = np.sort(eigenvalue)[::-1]
    vt = np.transpose(eigenvector)
    sigmaValues = np.trim_zeros(getSigmaValues(eigenvalue))
    sigma = getSigma(matrix, sigmaValues)
    u = getU(matrix, eigenvector, sigmaValues)
    #u = getUWithNullSpace(matrix, ata, eigenvalue, sigmaValues)
    sigma = np.delete(sigma, np.s_[(len(u[0])):], 0)
    return u, sigma, vt



#mat = np.array([[3, 1, 1], [-1, 3, 1]])
#mat = np.array([[2, 2, 0], [-1, 1, 0]])
#mat = np.array([[4, 1, 3],[8, 3, -2]])
# mat = np.array([[2, 1, 0, 0],[4, 3, 0, 0]])
# mat = np.transpose(mat)
mat = readImage()
R = []
G = []
B = []
if(mat.ndim == 3):
    R = mat[:,:,0]
    G = mat[:,:,1]
    B = mat[:,:,2]
    ur, sigmar, vtr = svd(R)
    ug, sigmag, vtg = svd(G)
    ub, sigmab, vtb = svd(B)
    r = int(input("input r: "))
    recr = ur[:,:r].dot(sigmar[0:r, :r]).dot(vtr[:r, :])
    recg = ug[:,:r].dot(sigmag[0:r, :r]).dot(vtg[:r, :])
    recb = ub[:,:r].dot(sigmab[0:r, :r]).dot(vtb[:r, :])
    rec = np.zeros(mat.shape)
    print(rec.shape)
    rec[:,:,0] = recr
    rec[:,:,1] = recg
    rec[:,:,2] = recb
    resized = Image.fromarray(rec)
    #resized = resized.convert('RGB')
    resized.save("resized.png")
else:
    u, sigma, vt = svd(mat)
#u, sigma, vt = svd(mat)
#u, sigma, vt = svd(mat)
#sigma = np.diag(sigma)
# print("u = ", u)
# print("sigma = ", sigma)
# print("vt = ", vt)

    print("reconstructed size = ", u.dot(sigma).dot(vt).shape)
    r = int(input("input r: "))
    rec = u[:,:r].dot(sigma[0:r, :r]).dot(vt[:r, :])
    resized = Image.fromarray(rec)
    resized = resized.convert('RGB')
    resized.save("resized.png")
