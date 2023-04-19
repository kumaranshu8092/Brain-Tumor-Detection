# for reading image from the dataset
import cv2 
# for feeding tha dataset path to the cv2 library
import os
# for developing and training ML model
import tensorflow as tf
from tensorflow import keras
# for converting image from BGR to RGB
from PIL import Image
# for converting the image dataset to array
import numpy as np
# for spliting the dataset into train and test
from sklearn.model_selection import train_test_split
# for normalising the dataset
from keras.utils import normalize
# importing the Sequential model or neural network
from keras.models import Sequential
# Conv2D for generating the layers in our model
# MaxPooling is used to pool the max value from the layer in matrix
from keras.layers import Conv2D, MaxPooling2D
# Activation function converts the single perceptron to multilayer perceptron
# dropout is used to drop some features to avoid overfitting,unverfitting
# flatten is used to get the flatten array from matrix
# dense is outer layer of CNN which we can define apart from the CNN inner layers. 
from keras.layers import Activation, Dropout, Flatten, Dense
# A binary matrix representation of the input as a NumPy array
from keras.utils import to_categorical

image_directory='datasets/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')

# creating the two empty list
# dataset to store the images in array formate
# labels to store the tumor or non-tumor label in array formate
dataset=[]
label=[]

INPUT_SIZE=128

# created a loop in which we 1st:
# 1 - check if it is an image
# 2 - if it is an image then we will 
#       read it
#       convert it to RGB format from BGR formate as OpenCV reads images in BGR format and for image processing we need image in RGB format
#       we will resize it in 64,64
#       we added the image in dataset list in form of array
#       we added label for the image according to it's type(i.e. 1 for tumor images and 0 for non_tumor images)

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)


# we converted the list dataset to array 
dataset=np.array(dataset)
label=np.array(label)

# here we are splitting the data into 10% test and 90% train model 
# here x_train is training dataset
# x_test is testing dataset
# y_train is label of training dataset(i.e. 0 or 1)
# y_test is label of training dataset(i.e. 0 or 1)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.10, random_state=47)

# Reshape = (no of images, image_width, image_height, n_channel)

# we are normalizing the data here
x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

# we are defining the category to the labels and dividing it in two classes(i.e tumor and non tumor)
y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)



# Model Building
# 64,64,3

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Dropout(0.2))
model.add(Activation('softmax'))


# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax

# we are compiling the model that we have generated above
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# we are creating check point so that we can save the best model with the best accuracy automatically
from keras.callbacks import ModelCheckpoint, EarlyStopping
# model check point
mc = ModelCheckpoint(filepath="bestmodel.h5",monitor='Val_accuracy',verbose=1,save_best_only=True)
cb = [mc]
# we are training the model that we have created above

model.fit(x_train, y_train, 
batch_size=18, 
verbose=1, epochs=20, 
validation_data=(x_test, y_test),
shuffle=False,callbacks=cb)

# we are saving the above created deep learning model
model.save('bestmodel.hdf5')
