
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image_arr = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)

# Realizar la descomposición en valores singulares (SVD)
U, S, VT = np.linalg.svd(image_arr, full_matrices=False)

# Lista para almacenar las imágenes con diferentes cantidades de valores singulares
svd_images = []

traza_optima = sum(S)

k_optimo = 0
acercaminentos = []

flag = True
# Rango de valores singulares a considerar
for k in range(1, min(image_arr.shape)):
    s_temp = np.zeros(S.shape)
    for i in range(0,k):
        s_temp[i] = S[i]
    
    acercamiento = sum(s_temp)/ traza_optima
    
    reconstructed_image = U @ np.diag(s_temp) @ VT
    svd_images.append(reconstructed_image)
    
    acercaminentos.append(acercamiento)
    if (acercamiento >= 0.85 and flag):
        print("k óptimo desde el que el valor supera el 85% del cociente de la varianza explicada: ",k)
        k_optimo = k
        flag = False
        
plt.plot([i for i in range(0,len(acercaminentos))],acercaminentos)
plt.title("Cociente de varianza explicada VS # valores singulares")
plt.show()

plt.imshow(svd_images[k_optimo-1], cmap="gray")
plt.title(str(k_optimo) +" valores singulares")
plt.show()


for i, reconstructed_image in enumerate(svd_images):
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(str(i)+" valores singulares")
    plt.show()

plt.imshow(U @ np.diag(S) @ VT, cmap="gray")
plt.show()