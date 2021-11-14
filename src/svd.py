import numpy as np
import math
from PIL import Image
from scipy.linalg.decomp_svd import null_space
from matrix import *
from eigen import *
import time

def readImageFromFileName(filename):
    img = Image.open('src\\static\\uploads\\' + filename)
    imgFormat = img.format
    mode = input("For RGB, type 'RGB'. For grayscale type 'L': ")
    imgMatrix = np.array(img.convert(mode.upper()))
    return imgMatrix, imgFormat
def readImage(): #fungsi membaca image dari input user, mengembalikan matriks dari gambar dan format gambar
    filename = input("input file name (located in ../images/): ")
    img = Image.open('D:\\ProjectKuliah\\AlGeo\\Algeo02-20013\\src\\static\\images\\' + filename)
    imgFormat = img.format
    mode = input("For RGB, type 'RGB'. For grayscale type 'L': ")
    imgMatrix = np.array(img.convert(mode.upper()))
    return imgMatrix, imgFormat

def getSigma(matrixA, sigmaValues): #fungsi mendapat sigma dari nilai-nilai sigma yang diletakkan di diagonal
    x = np.array([[float(0) for j in range(len(matrixA[0]))] for i in range(len(matrixA))])
    
    for i in range(len(x)):
        if (i >= len(x[0])):
            break
        else:
            x[i][i] = sigmaValues[i]
    return x

def getSigmaValues(eigenvalues): #fungsi mendapatkan nilai-nilai sigma dari mengakarkan tiap eigenvalue
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
        u = np.append(u, np.matmul(tempU, ns), axis = 1)
    return(u)

def getU(matrix, eigenvector, sigmaValues): #ada cara lain dapat u ternyata
    u = []
    eigenVectorTransposed = np.transpose(eigenvector)
    for i in range(len(sigmaValues)):
        tempU = (1/sigmaValues[i])*matrix
        tempUTransposed = np.transpose(tempU)
        x = eigenVectorTransposed[i] @ tempUTransposed
        u.append(x)
    u = np.transpose(np.array(u))
    return(u)

def svd2(matrix): #cara lain mendapat svd, namun khusus untuk matrix persegi, tidak terpakai
    transposed = np.transpose(matrix)
    aat = matrix @ transposed
    
    eigenvalueaat, eigenvectoraat = eig(aat)
    eigenvalueaat = np.sort(eigenvalueaat)[::-1]
    sigmaValues = np.trim_zeros(getSigmaValues(eigenvalueaat))
    sigma = getSigma(matrix, sigmaValues)
    ata = transposed @ matrix 
    eigenvalueata, eigenvectorata = eig(ata)
    return eigenvectoraat, sigma, np.transpose(eigenvectorata)

def svd(matrix):
    print("transposing...")
    transposed = np.transpose(matrix)
    print("transposed")

    print("calculating ata...")
    ata = transposed @ matrix
    print("ata calculated")

    print("calculating eigen...")
    eigenvalue, eigenvector = eig(ata)
    print("eigen calculated")

    print("sorting eigenvalue...")
    eigenvalue = np.sort(eigenvalue)[::-1]
    print("eigenvalue sorted")

    print("calculating vt...")
    vt = np.transpose(eigenvector)
    print("vt calculated")

    print("calculating sigmavalues...")
    sigmaValues = np.trim_zeros(getSigmaValues(eigenvalue))
    print("sigmavalues calculated")

    # print("calculating sigma...")
    # sigma = getSigma(matrix, sigmaValues)
    # print("sigma calculated")

    print("calculating u...")
    # u = getU(matrix, eigenvector, sigmaValues)
    u = matrix @ eigenvector / sigmaValues #cara lain medapat u
    print("u calculated")

    # print("trimming sigma...")
    # sigma = np.delete(sigma, np.s_[(len(u[0])):], 0)
    # print("sigma trimmed")

    print("svd done!")
    return u, sigmaValues, vt

def getCompressed(matrix, r): #fungsi menerima matrix dan r (berapa nilai penting) dan mengembalikan hasil matriks yang sudah dikompresi serta byte untuk menyimpan u, sigma, dan vt nya
    u, sigma, vt = svd(matrix)
    sigma = np.diag(sigma)
    compressedU = u[:,:r]
    compressedSigma = sigma[0:r, :r]
    compressedVt = vt[:r, :]
    compressedBytes = sum([m.nbytes for m in [compressedU, compressedSigma, compressedVt]])
    fullSVDBytes = sum([m.nbytes for m in [u, sigma, vt]])
    return (compressedU @ compressedSigma @ compressedVt), compressedBytes, fullSVDBytes

def compress(filename, r):
    startTime = time.time()
    mat, imgFormat = readImageFromFileName(filename)
    print("output format will be:", imgFormat)
    imgShape = mat.shape
    print("image shape:", imgShape)
    r = int(input("input r (from 1 to " + (str(max(imgShape[0], imgShape[1]))) + "): "))
    compressedBytes = 0
    fullSVDBytes = 0
    if(mat.ndim == 3):
        rec = np.zeros(mat.shape)
        rec[:,:,0], cbr, fsvdbr = getCompressed(mat[:,:,0], r)
        rec[:,:,1], cbg, fsvdbg = getCompressed(mat[:,:,1], r)
        rec[:,:,2], cbb, fsvdbb = getCompressed(mat[:,:,2], r)
        compressedBytes = cbr+cbg+cbb
        fullSVDBytes = fsvdbr+fsvdbg+fsvdbb
        rec = rec.astype(np.uint8)
        resized = Image.fromarray(rec)
    else:
        rec, compressedBytes, fullSVDBytes = getCompressed(mat, r)
        rec = rec.astype(np.uint8)
        resized = Image.fromarray(rec).convert('RGB')

    resized.save(filename + "." + imgFormat.lower(), format = imgFormat.lower())
    originalBytes = mat.nbytes
    print("original bytes:", originalBytes, "\nfull svd bytes:", fullSVDBytes, "\ncompressed bytes: ", compressedBytes)
    compressionRate = compressedBytes*100/originalBytes
    print("compression rate ((compressed bytes)/(original bytes)):", str(compressionRate) + "%")
    runTime = startTime - time.time()
    return compressionRate, runTime


mat, imgFormat = readImage()
print("output format will be:", imgFormat)
imgShape = mat.shape
print("image shape:", imgShape)
r = int(input("input r (from 1 to " + (str(max(imgShape[0], imgShape[1]))) + "): "))
compressedBytes = 0
fullSVDBytes = 0
if(mat.ndim == 3):
    rec = np.zeros(mat.shape)
    rec[:,:,0], cbr, fsvdbr = getCompressed(mat[:,:,0], r)
    rec[:,:,1], cbg, fsvdbg = getCompressed(mat[:,:,1], r)
    rec[:,:,2], cbb, fsvdbb = getCompressed(mat[:,:,2], r)
    compressedBytes = cbr+cbg+cbb
    fullSVDBytes = fsvdbr+fsvdbg+fsvdbb
    rec = rec.astype(np.uint8)
    resized = Image.fromarray(rec)
else:
    rec, compressedBytes, fullSVDBytes = getCompressed(mat, r)
    rec = rec.astype(np.uint8)
    resized = Image.fromarray(rec).convert('RGB')

filename = input("input file name (without format): ")
resized.save(filename + "." + imgFormat.lower(), format = imgFormat.lower())
originalBytes = mat.nbytes
print("original bytes:", originalBytes, "\nfull svd bytes:", fullSVDBytes, "\ncompressed bytes: ", compressedBytes)
compressionRate = compressedBytes*100/originalBytes
print("compression rate ((compressed bytes)/(original bytes)):", str(compressionRate) + "%")