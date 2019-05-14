#import imutils
import cv2
from keras.models import load_model, Model
from keras.preprocessing.image import img_to_array
import numpy as np
import math

def modify_layer_row_to_zero(layer, filters, columns, rows, aisles):
  
  #dimensions of layer:
  dim1 = len(layer.get_weights())
  dim2 = len(layer.get_weights()[0])           #dim of filters column
  dim3 = len(layer.get_weights()[0][0])        #dim of filters row
  dim4 = len(layer.get_weights()[0][0][0])     #dim of filters aisle
  dim5 = len(layer.get_weights()[0][0][0][0])  #number of filters filter
  zeros_row=np.zeros([1,dim5])
  layer_weight=layer.get_weights() 
  for f in filters:
    if f < dim5:
      for c in columns:
        if c<dim2:
        
          for r in rows:
            if r<dim3:
            
              for a in aisles:
                if a<dim4:
                 layer_weight[0][c][r][a][f]=0;
                 layer.set_weights(layer_weight)
                 print("Modified",layer.name,c,r,a,f)
                else:
                  print("Layer : ",layer.name, " column :" ,c,"row :", r,"Has less then ",a, " elements" )
                
            else:
              print("Layer : ",layer.name, " column :", c,"Has less then ",r, " rows" )
        else:
          print("Layer : ",layer.name, "Has less then ",c, " columns." )
    else:      
      print("Layer : ", "Has less then ",f, " filters." )
                

def set_layer_to_zero(layer):
  dim1 = len(layer.get_weights())#rien
  dim2 = len(layer.get_weights()[0])
  dim3 = len(layer.get_weights()[0][0])  
  dim4 = len(layer.get_weights()[0][0][0])  
  dim5 = len(layer.get_weights()[0][0][0][0])  
  l=[]#Each layer is a list
  new_layer=np.ones([dim2,dim3,dim4,dim5])#Each list has a array of 4 dimensions
  l.append(new_layer)#Add this array to the list
  layer.set_weights(l)#Set this list as weights of our arrays

def get_dims_conv(layers, layer_names):
 for (i,layer) in enumerate(layers): 
    if layer_names[i].startswith("conv") :
      dim1 = len(layer.get_weights())#rien
      dim2 = len(layer.get_weights()[0])
      dim3 = len(layer.get_weights()[0][0])  
      dim4 = len(layer.get_weights()[0][0][0])  
      dim5 = len(layer.get_weights()[0][0][0][0])  
      print(layer_names[i]," has : \n","ellements",dim1,"filters:",dim5,"columns: ", dim2,"rows: ",dim3,"aisles: ",dim4)



def show_layers (input,layer_name_to_show,name) :
  layer_outputs = [layer.output for layer in emotion_classifier.layers[1:len(emotion_classifier.layers)-2]]
  activation_model = Model(inputs=emotion_classifier.input, outputs=layer_outputs)
  activations = activation_model.predict(input)
  images_per_row = 16
  for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    if layer_name.startswith(layer_name_to_show):
      n_features = layer_activation.shape[-1] # Number of features in the feature map
      size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
      n_cols = images_per_row # Tiles the activation channels in this matrix
      n_rows = math.ceil(n_features / n_cols)
      # dtype of the numpy array has to be uint8. Else cv2.imshow fails
      display_grid = np.zeros((size * n_rows, size * n_cols), dtype='uint8')
      #_min = layer_activation[0, :, :, :].min()
      #_max = layer_activation[0, :, :, :].max()
      for row in range(n_rows): # Tiles each filter into a big horizontal grid
          for col in range(n_cols):
              feature_id = row * n_cols + col
              if feature_id < n_features:
                  channel_image = layer_activation[0, :, :, feature_id]
                  _min = channel_image.min()
                  _max = channel_image.max()
                  if _max != _min:
                      # normalize float values as unsigned int 8 [0-255]
                      channel_image -= _min
                      channel_image /= (_max-_min)
                      channel_image *= 255
                      channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                      # put the image in the right position in th edisplay grid
                      display_grid[row * size : (row + 1) * size,
                                   col * size : (col + 1) * size] = channel_image
      # sclae the image to 64x64
      scaled_display = cv2.resize(display_grid,(n_cols*64, n_rows*64))
      #cv2.imshow(layer_name, scaled_display)
      cv2.imwrite("compare/"+layer_name+"_"+name+".png", scaled_display)
  # hit any key to close all the windows

  #k = cv2.waitKey(0) & 0xFF
  #cv2.destroyAllWindows()



def set_layer_name_to_zero (layers, layer_names, layer_name): 
  for (i,layer) in enumerate(layers): 
    if layer_names[i].startswith(layer_name) :
      set_layer_to_zero(layer)
    
def set_layer_name_row_to_zero (layers, layer_names, layer_name, filters, columns, rows, aisles): 
  for (i,layer) in enumerate(layers): 
    if layer_names[i].startswith(layer_name) :
      modify_layer_row_to_zero(layer, filters, columns, rows, aisles)
      
def predict_emotion(emotion_classifier,input):
  preds = emotion_classifier.predict(input)[0]
  emotion_probability = np.max(preds)
  label = EMOTIONS[preds.argmax()]
  print("predict: {}, {:2.2}".format(label, emotion_probability))
      
#loading model
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

#loiding image
input = cv2.imread('img2.png') # a 64x64x3 gray image
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
#activations = activation_model.predict(input)

#activation_model.summary()

#tmp = activation_model.get_config()


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

##########################################################################
get_dims_conv(layers, layer_names)
 
predict_emotion(emotion_classifier,input)
show_layers (input,"conv2d_1","before")

#
layer="conv2d_4"
modification_filters=[range(0,100)]
columns={0,1,2,3,4}
rows={0}
aisles={0}
for filters in modification_filters: 
  for i in range(0,7):
    emotion_classifier = load_model(emotion_model_path, compile=False)
    layer_outputs = [layer.output for layer in emotion_classifier.layers[1:len(emotion_classifier.layers)-2]]
    activation_model = Model(inputs=emotion_classifier.input, outputs=layer_outputs)
    layers=[]
    for name in layer_names:
      layers.append(activation_model.get_layer(name))
  
  
    layer="conv2d_"+str(i)
    show_layers (input,layer,"before")
    set_layer_name_row_to_zero(layers, layer_names, layer, filters, columns, rows, aisles)
  #set_layer_name_to_zero (layers, layer_names, layer)
    predict_emotion(emotion_classifier,input)
    show_layers (input,layer,"after")

      
      
      
      
      
      
      
