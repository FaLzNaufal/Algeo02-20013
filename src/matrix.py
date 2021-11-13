
import numpy as np
from scipy.linalg import hessenberg
import sympy as sy

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
    if n == 2 :
        temp = temp.tolist()
        linier = temp[0][0] + temp[1][1]
        linier *= -1
        const = temp[0][0] * temp[1][1] - temp[0][1] * temp[1][0]
        coeff = [1, linier, const]
        result = np.roots(coeff)
    else :
        temp = toHessenberg(temp)
        for k in range(nIteration): 
            s = temp.item(n-1, n-1) 
            smult = s * np.eye(n)
            Q, R = np.linalg.qr(np.subtract(temp, smult))
            temp = np.add(R @ Q, smult)

        result = [0 for i in range(n)]
        for j in range(n) :
            result[j] = round(temp.item(j,j),5)
        result = sorted(result, reverse=True)

    return result # result berisi eigenvalue terurut mengecil

# return eselon baris tereduksi
def gaussJordan(matriks) :
    matriks, pivot = sy.Matrix(matriks).rref()
    matriks = np.array(matriks.tolist())
    return matriks, pivot

def eigenVector(matriks, lamda) :
    temp = np.copy(matriks)
    I = np.eye(len(matriks)) * lamda
    temp = np.subtract(temp, I)

    mat, pivot = gaussJordan(temp)
    mat = mat.tolist()
    res = createMatrix(len(matriks),1)
    for col in range(len(mat[0])) :
        if (col not in pivot) :
            for row in range(len(mat)) :
                res[row][0] = -1 * mat[row][col]
            break

    n = 0
    for row in range(len(res)):
        if res[row][0] == 0 :
            n += 1

    firstZero = 0
    for row in range(len(res)) :
        if res[row][0] == 0 :
            firstZero = row
            break

    nRow = len(res)
    firstZero *= -1
    res = np.pad(res, ((0,0),(0,n-1)), mode='constant', constant_values=0)
    I = np.eye(nRow, n, k=firstZero)
    res = res + I
    res = res.tolist()

    return res

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


a = np.matrix([[10,0,2],[0,10,4],[2,4,2]])
b = eigenValues(a)
print(b)