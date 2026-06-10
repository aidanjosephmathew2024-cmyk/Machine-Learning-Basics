import keras
from keras.datasets import mnist
from keras.models import Sequential  #We can add how much ever layers we want
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

pixel_width = 28
pixel_height = 28
num_of_classes = 10

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
#RELU = Rectified linear unit

model.add(MaxPooling2D(pool_size=(2,2)))
print("Post MaxPooling ", model.output_shape)

#To avoide overfitting we use a Dropout layer
#It is going to randamize our data according to the percentage that we set
model.add(Dropout(0.25))
print("Post Dropout: ", model.output_shape)
