import numpy as np
from sympy import *

def eig(mat) :

    x = Symbol('x') # Anggap aja lambda
    I = (eye(shape(mat)[0]) * x) - mat
    I = I.det()

    result = list(solveset(I,x)) # berisi eigen value, tapi masih dalam bentuk Sympy

    arr_of_res = []
    for item in result :
        matCopy = np.copy(mat)
        identity = eye(shape(mat)[0]) * item
        identity -= matCopy

        gauss, pivot = identity.rref() # Masih harus dicari nilai x1, x2, dan x3

        gauss = np.array(gauss) # biar bisa dapat len(gauss[0])

        n = len(gauss) - len(pivot)
        res = np.zeros((len(gauss), n))
        for col in range(len(gauss)) :
            if (col not in pivot) :
                i = 0
                for row in range(len(gauss)) :
                    res[row][i] = -1 * gauss[row][col]
                res[col][i] = 1
                i += 1

        sum = 0
        for item in res :
            sum += item[0] * item[0]
        sum = sum ** 0.5

        for i in range(len(res)) :
            res[i] = res[i][0] / sum

        res = res.tolist() # isinya masih agak jelek

        res1= []
        for item in res :
            res1.append(item[0])

        arr_of_res.append(res1)
        
    result_vector = np.matrix(arr_of_res)
    result_vector = result_vector.T

    for i in range(len(result)) :
        result[i] = float(result[i].evalf()) # sekarang result dikonversi ke float

    return result, result_vector

# mat = np.array([[10,0,2],[0,10,4],[2,4,2]])
# a, b = eig(mat)
# print(a)
# print(b)