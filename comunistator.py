import cv2
import numpy as np
from fastai.vision import *

#cargamos el modelo
learn = load_learner('./')

#definimos la función de calculo de coordenadas del centro de la cara
def cmp_ctr(img):
    ctr = learn.predict(img)
    c1 = (1 + ctr[0].data[0][0]) * img.size[0]/2
    c2 = (1 + ctr[0].data[0][1])* img.size[1]/2
    return tensor([c1,c2])

#cargamos la imagen del camarada
img_stalin = cv2.imread('stalin.jpg')

#capturamos la camara
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        #convertimos el frame a un objeto Image de fastai y calculamos las coordenadas del centro
        t = torch.tensor(np.ascontiguousarray(np.flip(frame, 2)).transpose(2,0,1)).float()/255
        img_test = Image(t)
        coords = cmp_ctr(img_test)

        #cambiamos los píxeles de la cara por la foto del camarada
        frame[coords[0].int() - 140: coords[0].int()+140, coords[1].int() - 100 : coords[1].int() + 100] = img_stalin

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
