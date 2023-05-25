# Roy Afaryan
# CS3310
# Project 1

import time
import numpy as np
import random
from statistics import mean

# algorithm for classical matrix multiplication
def classical(a_matrix, b_matrix):
    
    # n = size of matrix
    n = len(a_matrix)
    
    # result matrix
    c_matrix = np.array([[0 for i in range(n)] for j in range(n)])


    # nested for loops to calculate result matrix
    for i in range(n):
        
        for j in range(n):
            c_matrix[i][j] = 0
            
            for k in range(n):
                c_matrix[i][j] += a_matrix[i][k] * b_matrix[k][j]

    # return result matrix
    return c_matrix
                

# algorithm for standard divide and conquer matrix multiplication
def divide_and_conquer(a_matrix, b_matrix):

    # n = size of matrix
    n = int(len(a_matrix))
    
    # result matrix
    c_matrix = np.array([[0 for i in range(n)] for j in range(n)])
    
    # base case
    if n == 1:
        c_matrix[0][0] = a_matrix[0][0] * b_matrix[0][0]
    else:
        # divide a_matrix into smaller submatrices
        a11 = a_matrix[:n//2, :n//2]
        a12 = a_matrix[:n//2, n//2:]
        a21 = a_matrix[n//2:, :n//2]
        a22 = a_matrix[n//2:, n//2:]
        
        # divide b_matrix into smaller submatrices
        b11 = b_matrix[:n//2, :n//2]
        b12 = b_matrix[:n//2, n//2:]
        b21 = b_matrix[n//2:, :n//2]
        b22 = b_matrix[n//2:, n//2:]

        # recursive call
        c11 = divide_and_conquer(a11, b11) + divide_and_conquer(a12, b21)
        c12 = divide_and_conquer(a11, b12) + divide_and_conquer(a12, b22)
        c21 = divide_and_conquer(a21, b11) + divide_and_conquer(a22, b21)
        c22 = divide_and_conquer(a21, b12) + divide_and_conquer(a22, b22)

        # combine step
        c_matrix[:n//2, :n//2] = c11
        c_matrix[:n//2, n//2:] = c12
        c_matrix[n//2:, :n//2] = c21
        c_matrix[n//2:, n//2:] = c22

    # return result matrix
    return c_matrix

# algorithm for strassen's matrix multiplication
def strassen(a_matrix, b_matrix):

    # n = size of matrix
    n = int(len(a_matrix))
    
    # result matrix
    c_matrix = np.array([[0 for i in range(n)] for j in range(n)])

    # base case
    if n == 1:
        c_matrix[0][0] = a_matrix[0][0] * b_matrix[0][0]
    else:
        # divide a_matrix into smaller submatrices
        a11 = a_matrix[:n//2, :n//2]
        a12 = a_matrix[:n//2, n//2:]
        a21 = a_matrix[n//2:, :n//2]
        a22 = a_matrix[n//2:, n//2:]
        
        # divide b_matrix into smaller submatrices
        b11 = b_matrix[:n//2, :n//2]
        b12 = b_matrix[:n//2, n//2:]
        b21 = b_matrix[n//2:, :n//2]
        b22 = b_matrix[n//2:, n//2:]

        # recursive call using strassen's formula
        p = strassen(a11 + a22, b11 + b22)
        q = strassen(a21 + a22, b11)
        r = strassen(a11, b12 - b22)
        s = strassen(a22, b21 - b11)
        t = strassen(a11 + a12, b22)
        u = strassen(a21 - a11, b11 + b12)
        v = strassen(a12 - a22, b21 + b22)

        # combine step
        c11 = p + s - t + v
        c12 = r + t
        c21 = q + s
        c22 = p + r - q + u

        c_matrix[:n//2, :n//2] = c11
        c_matrix[:n//2, n//2:] = c12
        c_matrix[n//2:, :n//2] = c21
        c_matrix[n//2:, n//2:] = c22

    # return result
    return c_matrix




def main():

    global size 
    size = int(input("Enter size: "))
    CMM = []
    DCMM = []
    STRA = []

    for i in range(10):
        a_matrix_1 = np.array([[random.randint(0, 10) for i in range(size)] for j in range(size)])
        b_matrix_1 = np.array([[random.randint(0, 10) for i in range(size)] for j in range(size)])
        
        print("Matrix A:")
        print(a_matrix_1)

        print("Matrix B:")
        print(b_matrix_1,"\n")


        start_time = time.time()
        print("Classical MM:")
        print(classical(a_matrix_1, b_matrix_1))
        end_time = (round(time.time() - start_time, 5))
        print("--- %s seconds ---" % end_time)
        CMM.append(end_time)

        start_time = time.time()
        print("\nDivide and Conquer MM:")
        print(divide_and_conquer(a_matrix_1, b_matrix_1))
        end_time = (round(time.time() - start_time, 5))
        print("--- %s seconds ---" % end_time)
        DCMM.append(end_time)

        start_time = time.time()
        print("\nStrassen MM:") 
        print(strassen(a_matrix_1, b_matrix_1))
        end_time = (round(time.time() - start_time, 5))
        print("--- %s seconds ---" % end_time)
        STRA.append(end_time)
        
    print()
    print("Average Time of Classical MM: ", mean(CMM))
    print("Average Time of Divide-and-Conquer MM: ", mean(DCMM))
    print("Average Time of Strassen's MM: ", mean(STRA))
    
if __name__ == "__main__":    
    main()