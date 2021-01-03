from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import csv
import cv2
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

def CNN_Arch(image_size):

    model = Sequential()
    model.add(Conv2D(32, (3,3), strides=(1,1), activation='relu', input_shape=image_size))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(32, (3,3), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(32, (3,3), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

input_file = open('training_data.csv','r')
reader = csv.reader(input_file)
x_data = []
y_data = []

zero = 0
one = 0
for row in reader:
    image_name = row[0]
    #print(image_name)
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r,c = img.shape
    imgray = img.reshape([r, c, 1])
    count = 0
    x_data.append(imgray)
    y_data.append(int(row[1]))
    if int(row[1]) == 0:
        zero += 1
    else:
        one += 1

#callbacks we have to define
print(zero,one)
print("training data loaded.")

x_test = []
y_test = []

for root,dirs,files in os.walk('./normalsVsAbnormalsV1'):
    for file in files:
        image_name = os.path.join(root,file)
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(190,190))
        img = img.reshape([190,190,1])
        x_test.append(img)
        if "abnormalsJPG" in image_name:
            y_test.append(1)
        else:
            y_test.append(0)

x_data = np.array(x_data)
y_data = np.array(y_data)
x_test = np.array(x_test)
y_test - np.array(y_test)

BATCH_SIZE = 256
image_size = x_data[0].shape
#print(image_size)
model = CNN_Arch(image_size)

EPOCHS = 50

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='auto')
callbacks = [early_stopping]

filepath = "./model_cnn_Avi/cnn.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(x_data,y_data, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1,validation_data=(x_test,y_test))

model.save_weights("CNN_Model_Arch1.h5")