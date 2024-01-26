import cv2
import os
import numpy as np

dataPath = "C:/Users/danie/OneDrive/Documentos/ReconocimientoFacial/Data"
peopleList = os.listdir(dataPath)
print("lista de personas: ", peopleList)

#creamos arrays para almecenar las eticketas y las imagenes del rostor
labels = []
facesData = []
#se crea label para saber como indice al contador
label = 0

#se crean eticketas para identidicar a quien le pertenecen las imagenes
for nameDir in peopleList:
  personPath = dataPath + "/" + nameDir
  print("Leyenda los imagenes")

#se reconocen las imagenes y se transforman a la escala de grises
  for fileName in os.listdir(personPath):
    print("Rostros: ", nameDir + "/" + fileName)
    labels.append(label)
    facesData.append(cv2.imread(personPath + "/" + fileName,0))
    image = cv2.imread(personPath + "/" + fileName,0)
  label = label + 1

cv2.destroyAllWindows()

#imprimimos las etiquetas de imagen para verificar cuantas hay
print("labels = ",labels)
print("numero de etiquetas daniel: ", np.count_nonzero(np.array(labels)==0))
#print("numero de etiquetas andres: ", np.count_nonzero(np.array(labels)==1

#cramos una variable con el metodo de entrenamiento que se usara que es el EigenFaceRecognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
print("Entrenando...")

#comenzamos el entramiento, ingresando como metodos el array donde estan las fotos y sus etiquetas 
face_recognizer.train(facesData, np.array(labels))

#guardamos el modelo
face_recognizer.write("modeloface.xml")
print("Modelo almacenado...")
