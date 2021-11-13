import numpy as np
import math
from PIL import Image
from scipy.linalg.decomp_svd import null_space
from matrix import *
from numpy import linalg as LA

def readImage():
    filename = input("input file name (located in ../images/): ")
    img = Image.open('D:\\ProjectKuliah\\AlGeo\\Algeo02-20013\\src\\static\\images\\' + filename)
    imgFormat = img.format
    #imgMatrix = np.array(img.convert("L"))
    imgMatrix = np.array(img.convert("RGB"))
    #imgMatrix = np.asarray(img).astype(np.uint8)
    return imgMatrix, imgFormat

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
    for i in range(len(sigmaValues)):
        tempU = (1/sigmaValues[i])*matrix
        tempUTransposed = np.transpose(tempU)
        x = np.matmul(eigenVectorTransposed[i], tempUTransposed)
        u.append(x)
    u = np.transpose(np.array(u))
    return(u)

def svd2(matrix): #ini gajadi
    transposed = np.transpose(matrix)
    aat = np.matmul(matrix, transposed)
    
    eigenvalueaat, eigenvectoraat = np.linalg.eig(aat)
    eigenvalueaat = np.sort(eigenvalueaat)[::-1]
    sigmaValues = np.trim_zeros(getSigmaValues(eigenvalueaat))
    sigma = getSigma(matrix, sigmaValues)
    # squaredsigma = []
    # if len(sigma) > len(sigma[0]):
    #     squaredsigma = np.delete(sigma, np.s_[len(sigma[0]):], 0)
    # else:
    #     squaredsigma = np.delete(sigma, np.s_[len(sigma):], 1)
    # vt = np.linalg.inv(squaredsigma) @ np.transpose(eigenvectoraat) @ matrix
    ata = np.matmul(transposed, matrix)
    eigenvalueata, eigenvectorata = np.linalg.eig(ata)

    return eigenvectoraat, sigma, np.transpose(eigenvectorata)

def svd(matrix):
    print("transposing...")
    transposed = np.transpose(matrix)
    print("transposed")
    print("calculating ata...")
    ata = transposed @ matrix
    print("ata calculated")
    print("calculating eigen...")
    eigenvalue, eigenvector = np.linalg.eig(ata)
    #eigenvalue, eigenvector = eigen(ata)
    print("eigen calculated")
    print("converting eigenvalue to real...")
    eigenvalue = np.real(eigenvalue)
    print("eigenvalue converted to real")
    print("converting eigenvector to real...")
    eigenvector =  np.real(eigenvector)
    print("eigenvector converted to real")
    print("sorting eigenvalue...")
    eigenvalue = np.sort(eigenvalue)[::-1]
    print("eigenvalue sorted")
    print("calculating vt...")
    vt = np.transpose(eigenvector)
    print("vt calculated")
    print("calculating sigmavalues...")
    sigmaValues = np.trim_zeros(getSigmaValues(eigenvalue))
    print("sigmavalues calculated")
    print("calculating sigma...")
    sigma = getSigma(matrix, sigmaValues)
    print("sigma calculated")
    print("calculating u...")
    u = getU(matrix, eigenvector, sigmaValues)
    print("u calculated")
    print("trimming sigma...")
    sigma = np.delete(sigma, np.s_[(len(u[0])):], 0)
    print("sigma trimmed")
    print("svd done!")
    return u, sigma, vt



#mat = np.array([[3, 1, 1], [-1, 3, 1]])
#mat = np.array([[2, 2, 0], [-1, 1, 0]])
#mat = np.array([[4, 1, 3],[8, 3, -2]])
#mat = np.array([[2, 1, 0, 0],[4, 3, 0, 0]])
#mat = np.transpose(mat)
mat, imgFormat = readImage()
print(imgFormat)
print("image shape = ", mat.shape)
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
    rec[:,:,0] = recr
    rec[:,:,1] = recg
    rec[:,:,2] = recb
    rec = rec.astype(np.uint8)
    print(rec)
    resized = Image.fromarray(rec)
    filename = input("input file name: ")
    resized.save(filename + "." + imgFormat.lower(), format = imgFormat.lower())
else:
    u, sigma, vt = svd(mat)
    # print("u = ", u)
    # print("sigma = ", sigma)
    # print("vt = ", vt)
    # print("reconstructed = ", u.dot(sigma).dot(vt))
    r = int(input("input r: "))
    rec = u[:,:r].dot(sigma[0:r, :r]).dot(vt[:r, :])
    rec = rec.astype(np.uint8)
    resized = Image.fromarray(rec)
    resized = resized.convert('RGB')
    filename = input("input file name: ")
    filename = "resized"
    resized.save(filename + "." + imgFormat.lower(), format = imgFormat.lower())
