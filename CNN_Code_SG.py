#Python program to run the convolutional neural network for classyfing images of landscapes into 6 categories
import scipy as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


trainpath = "C:\Users\SUCHETA GHOSH\OneDrive\Desktop\ml assignment\ImageClassification\Train"
valpath = "C:\Users\SUCHETA GHOSH\OneDrive\Desktop\ml assignment\ImageClassification\Validation"
predpath = "C:\Users\SUCHETA GHOSH\OneDrive\Desktop\ml assignment\ImageClassification\Prediction"


Data = ImageDataGenerator(rescale=1/255.0) 
train_Data = Data.flow_from_directory(trainpath,target_size=(150,150),shuffle=True,class_mode='categorical')
val_Data = Data.flow_from_directory(valpath,target_size=(150,150),class_mode='categorical')


#The model Architecture is made as: 3 Convolutional and 2 Fully connected layers
pool=layers.MaxPooling2D((2,2))
in_layer=layers.Input(shape=(150,150,3))
con1=layers.Conv2D(32,(4,4),activation='relu',kernel_regularizer='l2')
con2=layers.Conv2D(32,(4,4),activation='relu',kernel_regularizer='l2')
flatten=layers.Flatten()

out_layer=layers.Dense(6,activation='softmax')
h1 = layers.Dense(100, activation='relu',kernel_regularizer='l2')
h2 = layers.Dense(100, activation='relu',kernel_regularizer='l2')
dropCon=layers.Dropout(0.2)
dropFNN=layers.Dropout(0.5)

#The convolutional layers
x=in_layer
x=con1(x)
x=dropCon(x)
x=pool(x)
x=con2(x)
x=dropCon(x)
x=pool(x)

x=flatten(x)

#The fully Connected Layers
x=h1(x)
x=dropFNN(x)
x=h2(x)
x=dropFNN(x)
outputs = out_layer(x)

CNN_Classifier = keras.Model(in_layer,outputs)

print(CNN_Classifier.summary())

CNN_Classifier.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adagrad(), metrics=['accuracy'])

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

#CNN_Data = CNN_Classifier.fit(train_Data, validation_data = val_Data, epochs=200, verbose=1, callbacks=[early])
CNN_Data = CNN_Classifier.fit(train_Data, validation_data = val_Data, epochs=200, verbose=1)

plt.plot(CNN_Data.history['accuracy'])
plt.plot(CNN_Data.history['val_accuracy'])
plt.title("Model accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

plt.plot(CNN_Data.history['loss'])
plt.plot(CNN_Data.history['val_loss'])
plt.title("Model loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()


reg.evaluate(test_Data, verbose = 1)
