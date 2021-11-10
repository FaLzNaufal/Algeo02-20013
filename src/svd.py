def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def multiply(matrix1, matrix2):
    return [[sum([matrix1[i][k]*matrix2[k][j] for k in range(len(matrix1[0]))]) for j in range(len(matrix2[0]))] for i in range(len(matrix1))]

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
    conjugated = [[matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]
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
    for r in getSolution(ata):
        print(r)



mat = [[0,0,1,4],
    [0,0,-2,8],
    [1,0,0,0]]
svd(mat)
