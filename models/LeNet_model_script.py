import numpy as np
from numpy import asarray
import math
import os
from PIL import Image
import pickle


# Hand-build LeNet model using weights from previously-trained model (using Keras)
# Start by building helper functions for each type of layer
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    X_pad = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    s = a_slice_prev * W
    Z = sum(sum(sum(s)))
    Z = Z + float(b)

    return Z


def conv_forward_initial(Img, W, b, hparameters):
    """
    Implements the forward propagation for the initial convolution function (taking the image as input)
    
    Arguments:
    Img -- set of images to be convolved
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from image set 
    (n_H_prev, n_W_prev, n_C_prev) = Img.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev - f + (2 * pad)) / stride) + 1
    n_W = int((n_W_prev - f + (2 * pad)) / stride) + 1

    # Initialize the output volume Z with zeros.
    Z = np.zeros((n_H, n_W, n_C))

    # Create padded image
    img_pad = zero_pad(Img, pad)

    for h in range(n_H):  # loop over vertical axis of the output volume
        # Find the vertical start and end of the current "slice" 
        vert_start = h * stride
        vert_end = (h * stride) + f

        for w in range(n_W):  # loop over horizontal axis of the output volume
            # Find the horizontal start and end of the current "slice"
            horiz_start = w * stride
            horiz_end = (w * stride) + f

            for c in range(n_C):  # loop over channels (= #filters) of the output volume

                # Use the corners to define the (3D) slice of the padded image 
                img_slice_pad = img_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                weights = W[:, :, :, c]
                biases = b[c]
                Z[h, w, c] = conv_single_step(img_slice_pad, weights, biases)

    return Z


def conv_forward_hidden(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape 
    (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev - f + (2 * pad)) / stride) + 1
    n_W = int((n_W_prev - f + (2 * pad)) / stride) + 1

    # Initialize the output volume Z with zeros.
    Z = np.zeros((n_H, n_W, n_C))

    # Create a_prev_pad by padding A_prev
    a_prev_pad = zero_pad(A_prev, pad)

    for h in range(n_H):  # loop over vertical axis of the output volume
        # Find the vertical start and end of the current "slice" 
        vert_start = h * stride
        vert_end = (h * stride) + f

        for w in range(n_W):  # loop over horizontal axis of the output volume
            # Find the horizontal start and end of the current "slice"
            horiz_start = w * stride
            horiz_end = (w * stride) + f

            for c in range(n_C):  # loop over channels (= #filters) of the output volume

                # Use the corners to define the (3D) slice of a_prev_pad 
                a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                weight = W[:, :, :, c]
                bias = b[c]
                Z[h, w, c] = conv_single_step(a_slice_prev, weight, bias)

    return Z


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """

    # Retrieve dimensions from the input shape
    (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A_pooled = np.zeros((n_H, n_W, n_C))

    for h in range(n_H):  # loop on the vertical axis of the output volume
        # Find the vertical start and end of the current "slice"
        vert_start = h * stride
        vert_end = (h * stride) + f

        for w in range(n_W):  # loop on the horizontal axis of the output volume
            # Find the vertical start and end of the current "slice"
            horiz_start = w * stride
            horiz_end = (w * stride) + f

            for c in range(n_C):  # loop over the channels of the output volume

                # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                a_prev_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                # Compute the pooling operation on the slice. 
                if mode == "max":
                    A_pooled[h, w, c] = np.max(a_prev_slice)
                elif mode == "average":
                    A_pooled[h, w, c] = np.mean(a_prev_slice)

    return A_pooled


def sigmoid(Z):
    return 1 / (1 + math.exp(-Z))


# Now get weights from previously-trained LeNet model
weights_file = open("/Users/alexk/PycharmProjects/WSSEF_Project/Parameters/weights.pkl", "rb")
weight_dict = pickle.load(weights_file)
weights_file.close()


# Compile helper functions into full LeNet model
def LeNet_forward_by_hand(image, weight_dict, pool_mode='max'):
    # Resize image
    image = image.resize((256, 256))

    # Rescale pixel values
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    img = pixels

    # First CONV layer
    hparameters = {}
    hparameters['pad'] = 2
    hparameters['stride'] = 1
    n_C = 20
    weights = weight_dict[1]
    biases = weight_dict[2]
    Z1 = conv_forward_initial(img, weights, biases, hparameters)

    # First ReLU layer
    A1 = np.maximum(Z1, 0)

    # First MAXPOOL layer
    hparameters['f'] = 2
    hparameters['stride'] = 2
    A1_pooled = pool_forward(A1, hparameters, pool_mode)

    # Second CONV layer
    hparameters['pad'] = 2
    hparameters['stride'] = 1
    n_C = 50
    weights = weight_dict[3]
    biases = weight_dict[4]
    Z2 = conv_forward_hidden(A1_pooled, weights, biases, hparameters)

    # Second ReLU layer
    A2 = np.maximum(Z2, 0)

    # Second MAXPOOL layer
    hparameters['f'] = 2
    hparameters['stride'] = 2
    A2_pooled = pool_forward(A2, hparameters, pool_mode)

    # Flatten layer
    A2_flat = A2_pooled.flatten()
    A2_flat = np.reshape(A2_flat, (-1, 1))

    # Densely connected layer with sigmoid activation function
    weights = weight_dict[5]
    biases = weight_dict[6]
    Z3 = np.dot(weights.T, A2_flat) + biases
    A3 = sigmoid(Z3)

    if A3 > 0.5:
        prediction = 'fungal'
        probability = A3
    elif A3 <= 0.5:
        prediction = 'bacterial'
        probability = 1 - A3

    print('predicted cause of infection: ' + prediction)
    print('estimated probability: %.2f' % (probability))

    return A3


# Run model on previously-unseen images (randomly selected)
image_name = input('What is the name of the input image file? (must be in current working directory)\n')
os.chdir("/Users/alexk/PycharmProjects/WSSEF_Project/Test_Images")
path = os.getcwd()
image = Image.open(path + '/' + image_name)

LeNet_forward_by_hand(image, weight_dict)
