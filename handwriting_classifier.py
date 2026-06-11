import keras
from keras.datasets import mnist
from keras.models import Sequential  #We can add how much ever layers we want
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pixel_width = 28
pixel_height = 28
num_of_classes = 10
batch_size = 32


(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

 
features_train = features_train.reshape(features_train.shape[0], 28, 28, 1)
features_test = features_test.reshape(features_test.shape[0], 28, 28, 1)

input_shape = (pixel_width, pixel_height, 1) #We are doing this for requirement of convolutional network

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

features_train /= 255
features_test /= 255         #Converting to percentages


labels_train = keras.utils.to_categorical(labels_train, num_of_classes)  #Here we take value of 2 turn to an array of 10 valueswhere the index that has to become 1 and the rest 0
labels_test = keras.utils.to_categorical(labels_test, num_of_classes)

#Convolutnal network creation starts
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))  #To add a layer complexity to the neural  network
print("Post Conv2D: ", model.output_shape)
#Here we will find 32 key features of the image that says it is that, so 32 filter layers
#kernal size how big of a grid should be on the image 
#RELU = Rectified linear unit to make sure there are no negative values

model.add(MaxPooling2D(pool_size=(2,2)))
#print("Post MaxPooling ", model.output_shape)
#Here we keep the most important pieces

#To avoide overfitting we use a Dropout layer
#It is going to randamize our data according to the percentage that we set
model.add(Dropout(0.25))
#print("Post Dropout: ", model.output_shape)
#We are not losing data, we are just performing convolusion

#Flatten
model.add(Flatten())
 #Now this is a single stack of 5408 nodes

#Now we connect all of these nodes to the next layer of nodes then add and divide there percentage
model.add(Dense(128, activation='relu'))  #Dense makes a fully connected layer
#128 nodes

#Now we pass to another fully connected layed and that is the output layer so we are going to use softmax activation
model.add(Dense(num_of_classes, activation='softmax'))

#Compiling of the model
model.compile(loss='categorical_crossentropy', 
              optimizer='adadelta',
              metrics=['accuracy'])
#keras.losses.categorical_crossentropy this categorieses the accurcy between the test and training data 

model.fit(features_train, labels_train,
          batch_size= batch_size, 
          epochs=10,
          verbose=1,
          validation_data=(features_test, labels_test))

score = model.evaluate(features_test, labels_test, verbose=0)

model.save(r'D:\Machine-Learning-Basics.keras') #.keras is how a keras file should be saved as\


import coremltools
coreml_model = coremltools.converters.keras.convert(model, input_names=['image'], image_input_names='image')

coreml_model.author = 'Devslopes, LLC.'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Predicts the handwritten characters passes in as a number between 1-9.'
coreml_model.input_description['image'] = 'A 28x28 pixel grayscale image.'
coreml_model.output_description['output1'] = 'A Multiarray where the index with the greatest float value (0-1) is the recognized digit.'

coreml_model.save(r'D:\Machine-Learning-Basics.mlmodel')




