import numpy as np


def gershgorin(A):
    λ_min, λ_max = 0,0

    # todo
    center_max, center_min = A[0][0], A[0][0]
    m = A.shape[0]
    for k in range(m):
        for l in range(m):
            if k == l:
                if A[k][k] > center_max:
                    center_max = A[k][k]

                if A[k][k] < center_min:
                    center_min = A[k][k]

                if (center_max == A[k][k]):
                    λ_max = np.copysign(np.sum(np.absolute(A[k][:])), center_max)

                if (center_min == A[k][k]):
                    if (center_min > 0):
                        λ_min = (2 * center_min) - np.sum(np.absolute(A[k][:]))
                    if (center_min < 0):
                        λ_min = np.copysign(np.sum(np.absolute(A[k][:])), center_min)
    return λ_min, λ_max

def power(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    # todo
    m = A.shape[0]
    # err.append(np.linalg.norm(np.matmul(A, np.transpose(v)) - (λ * v), np.inf))
    for k in iter(int, 1):
        w = A @ np.array(v).T
        v = w / np.linalg.norm(w)
        λ = np.matmul(np.matmul(np.transpose(v), A), v)
        err.append(np.linalg.norm((np.matmul(A, np.transpose(v)) - λ * v), np.inf))
        if err[k - 1] <= 10e-13:
            break
    return v, λ, err


def inverse(A, v0, μ):
    v = v0.copy()
    λ = 0
    err = []

    # todo
    m = A.shape[0]
    I = np.eye(m)
    # err = np.linalg.norm(np.matmul(A, v.T) - (λ * v.T), np.inf)
    # while err <= np.power(10,-13):
    for k in iter(int, 1):
        w = v @ np.linalg.inv(A - (μ * I))
        v = w / np.linalg.norm(w)
        λ = np.matmul(np.matmul(v.T, A), v)
        err.append(np.linalg.norm(np.matmul(A, v.T) - (λ * v.T), np.inf))
        if err[k - 1] <= 10e-13:
            break
    return v, λ, err


def rayleigh(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    # todo
    m = A.shape[0]
    λ = np.matmul(np.matmul(v.T, A), v)
    I = np.eye(m)
    for k in iter(int, 1):
        w = v @ np.linalg.inv(A - (λ) * I)
        v = w / np.linalg.norm(w)
        λ = np.matmul(np.matmul(v.T, A), v)
        err.append(np.linalg.norm(np.matmul(A, v.T) - (λ * v.T), np.inf))
        if err[k - 1] <= 10e-13:
            break
    return v, λ, err

def randomInput(m):
    #! DO NOT CHANGE THIS FUNCTION !#
    A = np.random.rand(m, m) - 0.5
    A += A.T  # make matrix symmetric
    v0 = np.random.rand(m) - 0.5
    v0 = v0 / np.linalg.norm(v0) # normalize vector
    return A, v0


if __name__ == '__main__':
    pass
    # todo
