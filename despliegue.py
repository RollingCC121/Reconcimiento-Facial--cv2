import cv2
import os

dataPath = "C:/Users/danie/OneDrive/Documentos/ReconocimientoFacial/Data"
imagePaths = os.listdir(dataPath)
print("imagePaths=", imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# se lee el modelo de deteccion que creamos 
face_recognizer.read("modeloEigenFace.xml")

#se especifica el video de prueba que vamos a usar 
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture("videos de Prueba/prueba2.mp4")

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
  ret, frame = cap.read()
  if ret == False: break
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  auxFrame = gray.copy()

  faces = faceClassif.detectMultiScale(gray,1.3,5)

  #se redimenciona la imagen del rostro a 150 pixeles
  for (x,y,w,h) in faces:
    rostro = auxFrame[y:y+h, x:x+w]
    rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
    result = face_recognizer.predict(rostro)#predecimos la etiqueta rostro y su confianza

    #generamos un rectangulo que muestre la confianza de la prediccion
    cv2.putText(frame, "{}".format(result),(x,y-5),1, .3,(255,255,0),1,cv2.LINE_AA)

  #la prediccion tiene que ser menor a 60 y entre mas cercano a cero mejor 
    if result[1] < 60:
        cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

  #si pasa de 60 no hay coincidencias
    else:
        cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

  cv2.imshow("frame",frame)
  k = cv2.waitKey(1)
  if k == 27:

cap.release()
cv2.destroyAllWindows()
