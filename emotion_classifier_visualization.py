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
print("---------------------------------------------------------------")
print(type(activations[4]))
print(activations[4][0][0])
print(np.zeros((1,8)))
for i in range(0,60):
 activations[4][0][i]=np.zeros((1,8))
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print(type(activation_model.get_layer("activation_1")))

print("---------------------------------------------------------------")
print("---------------------------------------------------------------")

print(activations[4])
layer_names = []
for layer in emotion_classifier.layers[1:len(emotion_classifier.layers)-2]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
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
    cv2.imshow(layer_name, scaled_display)

# hit any key to close all the windows
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

# pedict the emotion
preds = emotion_classifier.predict(input)[0]
emotion_probability = np.max(preds)
label = EMOTIONS[preds.argmax()]
print("predict: {}, {:2.2}".format(label, emotion_probability))
