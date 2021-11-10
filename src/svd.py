import numpy as np
from PIL import Image

def readImage():
    img = Image.open('D:\ProjectKuliah\AlGeo\Algeo02-20013\src\pepe.png')
    imgMatrix = np.array(img.convert('L'))   
    return imgMatrix

def transpose(matrix): #tidak dipakai, jadinya pakai fungsi bawaan numpy
    return np.array([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])

def multiply(matrix1, matrix2): #tidak dipakai, jadinya pakai fungsi bawaan numpy
    return np.array([[sum([matrix1[i][k]*matrix2[k][j] for k in range(len(matrix1[0]))]) for j in range(len(matrix2[0]))] for i in range(len(matrix1))])

def svd(matrix):
    transposed = np.transpose(matrix)
    ata = np.matmul(transposed, matrix)
    #fungsi eigen harusnya disini
    imgFromArray = Image.fromarray(matrix)
    imgFromArray.show()
    imgFromArray = Image.fromarray(transposed)
    imgFromArray.show()
    imgFromArray = Image.fromarray(ata)
    imgFromArray.show()


mat = readImage()
svd(mat)
