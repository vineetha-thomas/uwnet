from uwnet import *

def conv_net():
    l = [   
            # CNN
            make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10), 
            make_activation_layer(SOFTMAX)]


    return make_net(l)


def linear_net():
    l = [

            # FULLY CONNECTED
            make_connected_layer(32 * 32 * 3, 8 * 3 * 3 ), 
            make_activation_layer(LRELU),

            make_connected_layer(8 * 3 * 3 , 16 * 16 * 16), 
            make_activation_layer(LRELU),

            make_connected_layer( 16 * 16 * 16, 8 * 3 * 3), 
            make_activation_layer(LRELU),

            make_connected_layer(  8 * 3 * 3, 3628), 
            make_activation_layer(LRELU),

            make_connected_layer(  3628, 10), 
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
#m = linear_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))





# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

# Let's assume that we have fused multiply-adds so a matrix multiplication of a M x K matrix with 
# a K x N matrix takes M*K*N operations. How many operations does the convnet use during a forward pass?
# Conv layer1 : make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU) => 8 x (3x3x3) x (32x32) operations = 221,184
# Conv layer2 : make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU) => 16 x (3x3x8) x (16x16) operations = 294,912
# Conv layer3:  make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU) => 32 x (3x3x16) x (8x8) operations = 294,912
# Conv layer4:  make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU) => 64 x (3x3x32) x (4x4) operations = 294,912
# Final linear layer: make_connected_layer(256, 10) => 256*10 operations
# Total = 1,108,480 operations

# Designing a fully connected layer with same number of operations
# Layer 1: (32x32x3) - (8x3x3) => 221,184 operations
# Layer 2: (8x3x3) - (16x16x16) => 294,912 operations
# Layer 3: (16x16x16) - (8x3x3) => 294,912 operations
# Layer 4: (8x3x3) - (3628) => 261,216
# Layer 5: (3628) - (10) => 36280 operations
# Total = 1,108,504 operations

# Accuracy with the fully connected network:
# Training accuracy = 47.8%
# Test accuracy = 44.6%

# Accuracy with the original conv net:
# Training accuracy = 74.8%
# Test accuracy = 66.98%

# Hence we see that the training accuracy with a fully connected network is much lower when comapred to conv nets, even when both have
# similar number of neurons (or operations). One reason for this is the conv net architecture is better suited for images.
# For images, conv net architecture enables us to exploit the spatial information. To explain this, a pixel in an image is
# only related to its nearby pixels, and is not really realted to the pixels far away from it. A fully connected network will try to
# map the relation between each pixel to every other pixel, which is not desirable. Conv nets on the other hand considers this spatial
# relation between pixel as we are convolving the image with a filter of smaller size. 


