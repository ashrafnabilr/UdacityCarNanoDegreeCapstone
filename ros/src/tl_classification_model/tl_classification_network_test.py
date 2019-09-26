import csv
import cv2
import h5py
from keras.models import load_model


model = load_model("/home/ashre/model_sigmoid.h5")
image = cv2.imread('/home/ashre/LightStatusData/IMG/170.jpg')


print("Network prediction: ", int(model.predict(image[None,:,:,:], batch_size=1)))
# print("Ground Truth : ", int(-1))
