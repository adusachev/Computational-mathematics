import numpy as np

def check(A):
    """
    Проверяет условие сходимости метода Гаусса-Зейделя
    :param A: Матрица системы
    :return: None
    """
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    D = np.eye(len(A)) * np.diag(A)
    A_2 = - np.linalg.inv((L + D)) @ U
    q = np.linalg.norm(A_2, ord=2)
    assert q < 1, 'Метод не сходится'
    
    
def Gauss_Seidel_method(A, f, x0, eps=1e-6):
    """
    Решает СЛАУ Ax=f методом Гаусса-Зейделя
    :param A: матрица системы
    :param f: столбец свободных членов
    :param x0: начальное приближение
    :param eps: точность
    :return: вектор неизвестных x
    """
    # check(A)
    n = len(A)
    x1 = np.ones(n) * np.inf  # чтобы войти в цикл

    x = np.array([x0, x1])  # всегда храним только вектора на последних двух шагах
    k = 0
    x_tmp = x0  # вспомогательная, для контроля точности и выхода из цикла

    while np.linalg.norm(x[k+1] - x_tmp, ord=2) >= eps:
        for i in range(0, n):
            sum1 = 0
            sum2 = 0
            for j in range(0, i):
                sum1 += A[i, j] * x[k+1][j]
            for j in range(i+1, n):
                sum2 += A[i, j] * x[k][j]
            x[k+1][i] = (1.0 / A[i, i]) * (f[i] - sum1 - sum2)

        x_tmp = np.copy(x[k])
        x[k] = np.copy(x[k+1])

    return x[-1]

    
    
def test1():
    A = np.array([[2, 1], 
                  [1, 4]], dtype=float)
    f = np.array([1, 1], dtype=float)
    x0 = np.array([0, 0], dtype=float)
    x = Gauss_Seidel_method(A, f, x0)
    print('precise solution:')
    print('\t', [3/7, 1/7])
    print('numerical solution:')
    print('\t', x)


def test2():
    A = np.array([[4, 2, -1],
                  [2, 4, 1],
                  [-1, 1, 3]])
    f = np.array([1, 1, 1])
    x0 = np.array([0, 0, 0])
    x = Gauss_Seidel_method(A, f, x0)
    print('precise solution:')
    print('\t', [5/12, -1/12, 1/2])
    print('numerical solution:')
    print('\t', x)


def test3():
    A = np.array([[10, -1, 2, 0],
                  [-1, 11, -1, 3],
                  [2, -1, 10, -1],
                  [0, 3, -1, 8]])
    f = np.array([6, 25, -11, 15])
    x0 = np.array([0, 0, 0, 0])
    x = Gauss_Seidel_method(A, f, x0)
    print('precise solution:')
    print('\t', [1, 2, -1, 1])
    print('numerical solution:')
    print('\t', x)




if __name__ == '__main__':
    print('test 1:')
    test1()
    print()

    print('test 2:')
    test2()
    print()

    print('test 3:')
    test3()
