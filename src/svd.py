import numpy as np
import math
from PIL import Image
from matrix import *

def readImage():
    img = Image.open('D:\ProjectKuliah\AlGeo\Algeo02-20013\src\pepe.png')
    imgMatrix = np.array(img.convert('L'))   
    return imgMatrix

def transpose(matrix): #tidak dipakai, jadinya pakai fungsi bawaan numpy
    return np.array([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])

def multiply(matrix1, matrix2): #tidak dipakai, jadinya pakai fungsi bawaan numpy
    return np.array([[sum([matrix1[i][k]*matrix2[k][j] for k in range(len(matrix1[0]))]) for j in range(len(matrix2[0]))] for i in range(len(matrix1))])

def getSigma(matrixA, matrixAtA):
    x = np.array([[float(0) for j in range(len(matrixA[0]))] for i in range(len(matrixA))])
    sigmaValues = getSigmaValues(matrixAtA)
    for i in range(len(x)):
        if (i > len(x[0])):
            break
        x[i][i] = sigmaValues[i]
    return np.array(x)

def getSigmaValues(matrix):
    e = eigenValues(matrix)
    
    for i in range(len(e)):
        e[i] = math.sqrt(e[i])
    return e

def svd(matrix):
    transposed = np.transpose(matrix)
    ata = np.matmul(transposed, matrix)
    print(ata)
    eigenvalue, eigenvector = np.linalg.eig(ata)
    print(eigenvalue)
    print(eigenvector)
    sigma = getSigma(matrix, ata)
    #print(sigma)



mat = np.array([[2, 2, 0], [-1, 1, 0]])
#mat = readImage()
svd(mat)
