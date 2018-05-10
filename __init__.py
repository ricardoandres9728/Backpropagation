from model import Model

# import numpy as np
#
# inputs = [[0,0], [0,1], [1,0], [1,1]]
# outputs = [0, 1, 1, 0]
#
m = Model()
#
# m.train(inputs, outputs)
#
# for i in inputs:
#     p = m.predict(i)
#     print(str(i) + ' => ' + str(p))

from PIL import Image

imagen_entrenamiento = [Image.open('patrones/patron_entrena_e1.png'), Image.open('patrones/patron_entrena_e2.png'),
                        Image.open('patrones/patron_entrena_e3.png'), Image.open('patrones/patron_entrena_e4.png'),
                        Image.open('patrones/patron_entrena_e5.png'), Image.open('patrones/patron_entrena_a1.png'),
                        Image.open('patrones/patron_entrena_a2.png'), Image.open('patrones/patron_entrena_a3.png'),
                        Image.open('patrones/patron_entrena_a4.png'), Image.open('patrones/patron_entrena_a5.png')]
for i in range(0, len(imagen_entrenamiento)):
    imagen_entrenamiento[i].load()

imagen_prueba = [Image.open('patrones/patron_prueba_e1.png'), Image.open('patrones/patron_prueba_e2.png'),
                 Image.open('patrones/patron_prueba_a1.png'), Image.open('patrones/patron_entrena_a1.png')]

for i in range(0, len(imagen_prueba)):
    imagen_prueba[i].load()

conjunto_entrenamiento = [[], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
conjunto_prueba = []

for imagen in imagen_entrenamiento:
    datos_entrenamiento = []
    datos_entrenamiento2 = []
    height, weidht = imagen.size
    matriz = []
    blanco = (255, 255, 255)
    negro = (0, 0, 0)
    suma = 0
    for i in range(0, weidht):
        fila = []
        for j in range(0, height):
            if imagen.getpixel((j, i)) == negro:
                fila.append(1)
            else:
                fila.append(0)
        matriz.append(fila)

    # Promedios Filas
    for fila in matriz:
        datos_entrenamiento.append(sum(fila) / len(fila))
    # Promedios Columnas

    for i in range(0, 7):
        suma = 0
        for colum in matriz:
            suma += colum[i]
        datos_entrenamiento2.append(suma / 10)

    conjunto_entrenamiento[0].append(datos_entrenamiento + datos_entrenamiento2)
# Conjunto prueba _______________________________________________________________________
for imagen in imagen_prueba:
    datos_prueba = []
    datos_prueba2 = []
    height, weidht = imagen.size
    matriz = []
    blanco = (255, 255, 255)
    negro = (0, 0, 0)
    suma = 0
    for i in range(0, weidht):
        fila = []
        for j in range(0, height):
            if imagen.getpixel((j, i)) == negro:
                fila.append(1)
            else:
                fila.append(0)
        matriz.append(fila)

    # Promedios Filas
    for fila in matriz:
        datos_prueba.append(sum(fila) / len(fila))
    # Promedios Columnas

    for i in range(0, 7):
        suma = 0
        for colum in matriz:
            suma += colum[i]
        datos_prueba2.append(suma / 10)

    conjunto_prueba.append(datos_prueba + datos_prueba2)


m.train(conjunto_entrenamiento[0], conjunto_entrenamiento[1])

for i in conjunto_prueba:
    p = m.predict(i)
    print(str(i) + ' => ' + str(p))
