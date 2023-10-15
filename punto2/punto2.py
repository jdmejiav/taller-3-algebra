import numpy as np

A = np.array([[3, 2, -1],
              [3, -2, 0],
              [3, 2, 1],
              [3, -2, 0],
              [3, 2, -1]], dtype=np.float64)

m, n = A.shape
Q = np.zeros((m, n), dtype=np.float64)
R = np.zeros((n, n), dtype=np.float64)

for j in range(n):
    v = A[:, j].copy()
    for i in range(j):
        R[i, j] = np.dot(Q[:, i], A[:, j])
        v -= R[i, j] * Q[:, i]
    R[j, j] = np.linalg.norm(v)
    Q[:, j] = v / R[j, j]

print("Q:")
print(Q)
print("R:")
print(R)


Q_sci,R_sci = np.linalg.qr(A)

print("Con Scipy")

print("Q:")
print(Q_sci)
print("R:")
print(R_sci)

####
#### Punto 2. Cuando una matriz es linealmente dependiente, en las soluciones Q y R
#### Q puede tener múltiples soluciones o notener solución y R puede tener valores
#### nulos en la diagonal
####
#### Punto 3: Una matriz tiene factorización QR Única, cuando sus columnas son linealmente
#### independientes, y que sean de rango completo, es decir, de nulidad 0
####