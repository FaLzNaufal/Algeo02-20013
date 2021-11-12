
import numpy as np
from scipy.linalg import hessenberg

# Fungsi yang me-return matrix kosong berukuran nRow dan nCol
def createMatrix(nRow,nCol) :
    matrix = [[0 for col in range(nCol)] for row in range(nRow)]
    return matrix

# Prosedur untuk menampilkan matrix
def displayMatrix(matrix) :
    nRow = len(matrix)
    nCol = len(matrix[0])
    for row in range(nRow) :
        for col in range(nCol) :
            print(matrix[row][col], end=" ")
            if col == nRow - 1 :
                print("")

# Fungsi untuk menghitung determinan matrix
# Prekondisi : harus matrix persegi
def determinanMatrix(matrix) :
    matrixLength = len(matrix)
    if (matrixLength == 2) :
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    else :
        determinan = 0
        for col in range(matrixLength) :
            determinan += ((-1) ** col) * matrix[0][col] * determinanMatrix(partMatrix(matrix, 0, col))
        return determinan

# Fungsi yang me-return matrix jika dipartisi di baris nthRow dan kolom nthCol
def partMatrix(matrix, nthRow, nthCol) :
    nRow = nCol = len(matrix)
    newMatrix = [[0 for row in range(nRow - 1)] for col in range(nCol - 1)]
    idxRow = idxCol = 0
    for row in range(nRow) :
        for col in range(nCol) :
            if row != nthRow and col != nthCol :
                newMatrix[idxRow][idxCol] = matrix[row][col]
                if idxCol == len(newMatrix) - 1 :
                    idxRow += 1
                    idxCol = 0
                else :
                    idxCol += 1
    return newMatrix

# Fungsi yg me-return transpose matrix
def transposeMatrix(matrix) :
    nRow = len(matrix)
    nCol = len(matrix[0])
    transMat = createMatrix(nCol, nRow)
    for row in range(nRow) :
        for col in range(nCol) :
            transMat[col][row] = matrix[row][col]
    return transMat

# Return adjoin matriks
def adjMatrix(matrix) :
    length = len(matrix)
    temp = createMatrix(length, length)
    for row in range(length) :
        for col in range(length) :
            temp[row][col] =  ((-1) ** (col + row)) * determinanMatrix(partMatrix(matrix,row,col))
    temp = transposeMatrix(temp)
    return temp

# Return invers matriks
def inverseMatrix(matriks) :
    length = len(matriks)
    det = determinanMatrix(matriks)
    if det : # det != 0
        temp = adjMatrix(matriks)
        for row in range(length) :
            for col in range(length) :
                temp[row][col] /= det
    return temp

# Return nilai - nilai eigen
def eigenValues(matrix, nIteration = 50000) :
    temp = np.copy(matrix) 
    n = temp.shape[0] 
    I = np.eye(n) 
    for k in range(nIteration): 
        s = temp.item(n-1, n-1) 
        smult = s * np.eye(n)
        Q, R = np.linalg.qr(np.subtract(temp, smult))
        temp = np.add(R @ Q, smult)

    result = [0 for i in range(n)]
    for j in range(n) :
        result[j] = round(temp.item(j,j),2)
    result = sorted(result, reverse=True)

    return result # result berisi eigenvalue terurut mengecil

# Return array of eigenVector
def eigenVector(matriks) :
    temp = timesMatrix(matriks, inverseMatrix(getUpperTriangle(matriks)))
    for i in range(len(temp)) :
        for j in range(len(temp[0])) :
            temp[i][j] = round(temp[i][j], 2)
    return temp

def timesMatrix(matrixA, matrixB) : # matrix A x matrix B
    temp = np.dot(matrixA, matrixB)
    temp = temp.tolist()
    return temp

# Return segitiga atas dalam QR Decomposition
def getUpperTriangle(matrix) :
    r = np.linalg.qr(matrix, mode='r')
    r = r.tolist()
    lenght = len(r)
    for row in range(lenght) :
        for col in range(lenght) :
            r[row][col] = round(r[row][col], 2)
    return r

# Return Matriks Hessenberg
def toHessenberg(matriks) :
    temp = np.copy(matriks)
    temp = hessenberg(temp, overwrite_a=True)
    return temp

matriks = [[2.5, 1.1, 0.3], [2.2, 1.9, 0.4], [1.8, 0.1, 0.3]]
matriks = eigenVector(matriks)
print(matriks)


# a = np.matrix([[-26,-33,-25],[31,42,23],[-11,-15,-4]])   
# a = toHessenberg(a)

# print(a)
# b = eigenValues(a)

# print(b)
