#Python program to run the fully connected network for classyfing images of landscapes into 6 categories
import scipy as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


trainpath = "C:\Users\SUCHETA GHOSH\OneDrive\Desktop\ml assignment\ImageClassification\Train"
valpath = "C:\Users\SUCHETA GHOSH\OneDrive\Desktop\ml assignment\ImageClassification\Validation"
testpath = "C:\Users\SUCHETA GHOSH\OneDrive\Desktop\ml assignment\ImageClassification\Test"
predpath = "C:\Users\SUCHETA GHOSH\OneDrive\Desktop\ml assignment\ImageClassification\Prediction"

Data = ImageDataGenerator(rescale=1/255.0) 
train_Data = Data.flow_from_directory(trainpath,target_size=(150,150),shuffle=True,class_mode='categorical')
val_Data = Data.flow_from_directory(valpath,target_size=(150,150),shuffle=True,class_mode='categorical')
test_Data = Data.flow_from_directory(testpath,target_size=(150,150),shuffle=True, class_mode='categorical')


print(train_Data.class_indices)
#{'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5} (We have these categories in Test Data)

#The Model Architecture as built:2 hidden layers with 100 neurons each,output layer with 6 neurons for classification
#150x150 pixels
inputs = layers.Input(shape=(150,150,3))
flatten = layers.Flatten()
h1 = layers.Dense(100, activation='relu')
h2 = layers.Dense(100, activation='relu')
output = layers.Dense(6, activation='softmax')

       
x=flatten(inputs)
x=h1(x)
x=h2(x)
outputs=output(x)
unreg = keras.Model(inputs, outputs)
print(unreg.summary())

#We use categorical cross-entropy as a loss function for training
#Accuracy measures the frequency of matches between predictions and labels
unreg.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

UnregData = unreg.fit(train_Data, validation_data = val_Data, epochs=200, verbose=1)
plt.plot(UnregData.history['accuracy'])
plt.plot(UnregData.history['val_accuracy'])
plt.title("Unregularised Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

plt.plot(UnregData.history['loss'])
plt.plot(UnregData.history['val_loss'])
plt.title("Unregularised Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()


#Then we have this, modified network with l2 regularisation on weights
hreg1=layers.Dense(100,activation='relu',kernel_regularizer='l2')
hreg2=layers.Dense(100, activation='relu',  kernel_regularizer='l2')
dropout = layers.Dropout(0.5)
x=flatten(inputs)
x=hreg1(x)
x=dropout(x)
x=hreg2(x)
x=dropout(x)
outputs2=output(x)
reg = keras.Model(inputs, outputs2)

print(reg.summary())

reg.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adagrad(), metrics=['accuracy'])

regData = reg.fit(train_Data, validation_data = val_Data, epochs=200, verbose=1)
plt.plot(regData.history['accuracy'])
plt.plot(regData.history['val_accuracy'])
plt.title("Regularised Model accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

plt.plot(regData.history['loss'])
plt.plot(regData.history['val_loss'])
plt.title("Regularised Model loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()


#The gap between training and validation accuracy is significantly reduced here.
#l2 regularisation and dropout reduces overfitting and the model generalises better. 

#We stop the training if validation loss increases for 5 consecutive epochs. This is intended to improve generalisation.
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
DataStop1 = unreg.fit(train_Data, validation_data = val_Data, epochs=200, verbose=1, callbacks=[early]) 
 
plt.plot(DataStop1.history['accuracy'])
plt.plot(DataStop1.history['val_accuracy'])
plt.title("Early Stopping Accuracy (Unregularised)")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

plt.plot(DataStop1.history['loss'])
plt.plot(DataStop1.history['val_loss'])
plt.title("Early Stopping Loss (Unregularised)")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

#The accuracy on the validation set improves significantly with early stopping

DataStop2 = reg.fit(train_Data, validation_data = val_Data, epochs=200, verbose=1, callbacks=[early]) 
 
plt.plot(DataStop2.history['accuracy'])
plt.plot(DataStop2.history['val_accuracy'])
plt.title("Early Stopping Accuracy (Regularised)")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['training', 'validaiton'], loc='upper left')
plt.show()


plt.plot(DataStop2.history['loss'])
plt.plot(DataStop2.history['val_loss'])
plt.title("Early Stopping Loss (Regularised)")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

reg.evaluate(test_Data, verbose = 1)


