from PIL import Image,ImageOps
import numpy as np
import os
import csv
#Crea base de datos de imagenes generando el archivo X con las imagenes Y con la clase y P que indica a que persona esta asignado cada numero de clase
#La estructura de archivos es
"""
Raiz/
	imagenes/
		carpeta_con_clase_0/
		carpeta_con_clase_1/
				.
				.
	generar_base_datos.py
	----------------------
	#las respuestas se guardan en el mismo directorio del archivo .py
"""
def crear_base_datos(directorio,tam):
	lista_arrays_imgs=list()
	lista_clases_imgs=list()
	lista_persona_clase=list()
	personas=os.listdir(directorio)
	i=0
	for persona in personas:
		directorio_persona=os.path.join(directorio,persona)
		lista_imgs_personas=os.listdir(directorio_persona)
		for nombre_img in lista_imgs_personas:
			ruta_img=os.path.join(directorio_persona,nombre_img)
			img=Image.open(ruta_img)
			img = ImageOps.grayscale(img)
			img=img.resize(tam)
			img=np.asarray(img,dtype='float64')
			img=img.ravel()
			lista_arrays_imgs.append(img)
			lista_clases_imgs.append([i])
		lista_persona_clase.append([i,persona])
		i += 1
	X=np.array(lista_arrays_imgs)
	Y=np.array(lista_clases_imgs)
	P=np.array(lista_persona_clase)
	guardar_datos_csv(X,'X.csv')
	guardar_datos_csv(Y,'Y.csv')
	guardar_datos_csv(P,'P.csv')

#Guarda matrices en un archivo en formato csv
def guardar_datos_csv(X,ruta):
	with open(ruta, 'w', newline='', encoding='utf-8') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(X)

tam=(50,50)
directorio="imagenes"
crear_base_datos(directorio,tam)