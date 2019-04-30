#import imutils
import cv2
from keras.models import load_model, Model
from keras.preprocessing.image import img_to_array
import numpy as np
import math

#loading model
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


input = cv2.imread('img_sample.png') # a 64x64x3 gray image
input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY) # a 64x64x1 gray image
input = input.astype("float") / 255.0
input = img_to_array(input)
input = np.expand_dims(input, axis=0)

# get all layers output
# Creates a model that will return these outputs, given the model input
# we skip the first input, and the very last layers
layer_outputs = [layer.output for layer in emotion_classifier.layers[1:len(emotion_classifier.layers)-2]]
activation_model = Model(inputs=emotion_classifier.input, outputs=layer_outputs)

# Returns a list of Numpy arrays: one array per layer activation
activations = activation_model.predict(input)
activation_model.summary()
tmp = activation_model.get_config()



#model.layers[-1].activation = activations.example
#model.save(some_path)
#model = load_model(some_path)

layer_names=[]
for layer in emotion_classifier.layers[1:len(emotion_classifier.layers)-2]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot


#A list with all the layers of model
layers=[]
for name in layer_names:
 layers.append(activation_model.get_layer(name))

#Go through the list of all layers
for (i,layer) in enumerate(layers):
 if layer_names[i].startswith("conv") and layer_names[i] !="conv2d_7":
#if it is a convolution layer we will modify it. conv2d_7 has a two dmension list so didnt modify it.

  print(layer_names[i])#prints the names of the layers which will be modified.
#to get the dimension of uour layer. Probably a more simple method exists ;)

  dim1 = len(layer.get_weights())
  dim2 = len(layer.get_weights()[0])
  dim3 = len(layer.get_weights()[0][0])  
  dim4 = len(layer.get_weights()[0][0][0])  
  dim5 = len(layer.get_weights()[0][0][0][0])  

  l=[]#Each layer is a list
  new_layer=np.ones([dim2,dim3,dim4,dim5])#Each list has a array of 4 dimensions
  l.append(new_layer)#Add this array to the list
  layer.set_weights(l)#Set this list as weights of our arrays


# pedict the emotion
preds = emotion_classifier.predict(input)[0]
emotion_probability = np.max(preds)
label = EMOTIONS[preds.argmax()]
print("predict: {}, {:2.2}".format(label, emotion_probability))
