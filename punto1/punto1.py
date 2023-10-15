import numpy as np
import laguide as lag
A = np.array([[-2, 6, 2, -8],[-6, 0, 12, 12],[-6, 0, 12, 12],[-10, 3, 7, 14]])
X = np.array([[1],[0],[0],[0]])

m = 0
while (m < 20):
    X = A@X
    X = X/lag.Magnitude(X)
    m = m + 1
    
print(X)