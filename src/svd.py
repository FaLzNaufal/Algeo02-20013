import numpy as np
import math
from PIL import Image
from scipy.linalg.decomp_svd import null_space
from matrix import *

def readImage():
    img = Image.open('D:\ProjectKuliah\AlGeo\Algeo02-20013\src\pepe.png')
    imgMatrix = np.array(img.convert('L'))   
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
        x[i][i] = sigmaValues[i]
    return x

def getSigmaValues(eigenvalues):
    sigmaValues = np.array(eigenvalues, copy=True)
    for i in range(len(eigenvalues)):
        sigmaValues[i] = math.sqrt(abs(eigenvalues[i]))
    return sigmaValues

def getUWithNullSpace(matrixAtA, eigenValues, sigmaValues): #tidak terpakai
    u = np.array([])
    
    for i in range(len(eigenValues)):
        tmp = np.array(matrixAtA, copy=True)
        for j in range(len(tmp)):
            tmp[j][j] -= eigenValues[i]
        ns = null_space(tmp)
       
        #ns = ns * np.sign(ns[0,0])
        ns = np.transpose(ns)
        print("ns = ",ns)
        np.append(u, ns)
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

def svd(matrix):
    transposed = np.transpose(matrix)
    ata = np.matmul(transposed, matrix)
    eigenvalue, eigenvector = np.linalg.eig(ata)
    vt = np.transpose(eigenvector)
    sigmaValues = np.trim_zeros(getSigmaValues(eigenvalue))
    sigma = getSigma(matrix, sigmaValues)
    u = getU(matrix, eigenvector, sigmaValues)
    return u, sigma, vt



mat = np.array([[2, 2, 0], [-1, 1, 0]])
mat = np.transpose(mat)
#mat = readImage()
u, sigma, vt = np.linalg.svd(mat, full_matrices=False)
#u, sigma, vt = svd(mat)
sigma = np.diag(sigma)
print("u = ", u.shape)
print("sigma = ", sigma.shape)
print("vt = ", vt.shape)

print("rec = ", u.dot(sigma).dot(vt))
# r = int(input())
# rec = u[:,:r].dot(sigma[0:r, :r]).dot(vt[:r, :])
# resized = Image.fromarray(rec)
# resized.show()
