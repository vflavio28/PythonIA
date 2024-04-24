# Nome do aluno: Victor Flavio de Carvalho

# Resposta para COLOR_BGR2GRAY: ACHOU A PESSOA
# Resposta para COLOR_BGR2HSV: ACHOU A PESSOA
# Resposta para COLOR_BGR2LAB: ACHOU A PESSOA

import cv2 as tela

imagem = tela.imread('./cnh.png')

# Front face
df = tela.CascadeClassifier('./haarcascade_frontalface_default.xml')

faces = df.detectMultiScale(imagem, scaleFactor=1.10, minNeighbors=4, minSize=(35,35), flags=tela.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    tela.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 4)

tela.imshow("Original", imagem)
gray = tela.cvtColor(imagem, tela.COLOR_BGR2GRAY)
tela.imshow("Gray", gray)
hsv = tela.cvtColor(imagem, tela.COLOR_BGR2HSV)
tela.imshow("HSV", hsv)
lab = tela.cvtColor(imagem, tela.COLOR_BGR2LAB)


tela.imshow(str(len(faces)) + ' face(s) encontrada(s).', imagem)
tela.waitKey(0)
tela.destroyAllWindows()