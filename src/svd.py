import numpy as np
from PIL import Image

def readImage():
    img = Image.open('D:\ProjectKuliah\AlGeo\Algeo02-20013\src\pepe.png')
    imgMatrix = np.array(img.convert('L'))   
    return imgMatrix

def transpose(matrix):
    return np.array([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])

def multiply(matrix1, matrix2):
    return np.array([[sum([matrix1[i][k]*matrix2[k][j] for k in range(len(matrix1[0]))]) for j in range(len(matrix2[0]))] for i in range(len(matrix1))])

# def getEchelon(matrix):
#     echelon = matrix[:]
#     lastRow = 0
#     for i in range(len(echelon[0])):
#         for j in range(lastRow, len(echelon)):
#             if echelon[j][i] != 0:
#                 echelon[lastRow], echelon[j] = echelon[j], echelon[lastRow]
#                 lastRow =  lastRow + 1
#         if lastRow > len(echelon):
#             break
#     lastRow = 0
#     for i in range(len(echelon)):
#         for j in range(len(echelon[0])):
#             if echelon[i][j] == 1:
#                 break
#             if echelon[i][j] != 0:
#                 ratio = echelon[i][j]
#                 print(echelon[i][j],ratio)
#                 for k in range(j, len(echelon[i])):
#                     echelon[i][k] /= ratio
#                 break


#     return echelon


def getSolution(matrix):#matrixnya yg belum dikonjugasiin
    conjugated = np.array([[matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))], dtype='int16')
    for r in conjugated:
        r.append(0)
    for r in conjugated:
        print(r)
    print()
    solution = conjugated
    return solution

def svd(matrix):
    transposed = transpose(matrix)
    ata = multiply(transposed, matrix)
    #fungsi eigen harusnya disini
    imgFromArray = Image.fromarray(matrix)
    imgFromArray.show()
    imgFromArray = Image.fromarray(transposed)
    imgFromArray.show()
    imgFromArray = Image.fromarray(ata)
    imgFromArray.show()


mat = readImage()
svd(mat)
