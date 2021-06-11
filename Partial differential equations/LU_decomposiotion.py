
import numpy as np
import scipy.linalg



def LU_decomposition(a):
    n = len(a)

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sum1 = 0
            for k in range(i):
                sum1 += L[i, k] * U[k, j]

            if i <= j:
                U[i, j] = a[i, j] - sum1
            else:
                L[i, j] = (1 / U[j, j]) * (a[i, j] - sum1)

    return L, U





def test():
    a = np.array([[4, -1, -1, 0],
                  [-1, 4, 0, -1],
                  [-1, 0, 4, -1],
                  [0, -1, -1, 4]])
    p, L_s, U_s = scipy.linalg.lu(a)

    L, U = LU_decomposition(a)

    print(L_s)
    print(L)
    print()
    print(U_s)
    print(U)

    print()
    print(np.all(L == L_s), np.all(U == U_s))



if __name__ == '__main__':
    test()



















































