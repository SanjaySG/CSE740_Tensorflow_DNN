import tensorflow as tf
import numpy as np
import glob
import os
import pandas
from PIL import Image

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

global_image_dir = "./data/train/"
file_open = lambda x,y: glob.glob(os.path.join(x,y))
train_path_list = []
train_labels = []
test_path_list = []
test_labels = []
train_images = []

def get_file_names(row, path,label):
    image_path = global_image_dir+row[1]+"/"+row[2]
    path.append(image_path)
    label.append(row[1])

def image_process(row):
    image = Image.open(image_path)
    image = image.resize((320,240))
    #train_images.append(np.array(image))
    print image

    """
    image_path = global_image_dir+row[1]+"/"+row[2]
    print image_path
    image_queue = tf.train.string_input_producer([image_path])
    rteader = tf.WholeFileReader()
    key, value = reader.read(image_queue)
    current_img = tf.image.decode_jpeg(value)
    print current_img
    matrix_image(current_img)
    """

def matrix_image(image):
    image = list(image.getdata())
    image = Image.open(image)
    image = map(list,image)
    image = np.array(image)
    print image

def preprocess():
    data_df = pandas.read_csv('./data/driver_imgs_list.csv')
    print(data_df.columns)
    print data_df.shape
    trainSize = int(data_df.shape[0]*0.8)
    testSize = data_df.shape[0] - trainSize

    np_data = data_df.as_matrix()
    print "Data shape : "+str(np_data.shape)

    aperm = np.random.permutation(data_df.shape[0])
    train_data = np_data[aperm[0:trainSize],:]
    test_data = np_data[aperm[trainSize:],:]

    print "Train data shape : "+str(train_data.shape)
    print "Test data shape : "+str(test_data.shape)

    for row in train_data:
        get_file_names(row,train_path_list,train_labels)

#print "Train path list : "+str(train_path_list)
#print "Train labels : "+str(train_labels)
preprocess()

"""
Building a queue for sending the data
"""
filename_queue = tf.train.string_input_producer(train_path_list)
image_reader = tf.WholeFileReader()
img_key, image_tensor = image_reader.read(filename_queue)
image_tensor = tf.image.decode_jpeg(image_tensor,0,2)
#image_tensor = tf.image.resize_images(image_tensor,240, 320, method=0)
#label_tensor = tf.convert_to_tensor(train_labels)


"""
Model
"""
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

network = input_data(shape=[None, 240, 320, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=None)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with 10 outputss
network = fully_connected(network, 10, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='Image-classifier-initial.tfl.ckpt')
#

with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    for i in range(len(train_labels)):
        current_img = sess.run([image_tensor])
        print "Image tensor : "+str(current_img[0].shape)
        #print "Label : "+str(train_labels[i])
        model.fit({'input1': current_img[0]}, {'output1': train_labels[i]},show_metric=True)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)


print filename_queue.size()

"""
i = 0
for index, row in data_df.iterrows():
    print row['classname'],row['img']
    i+=1
    if i==10:
        break
"""




#train_images = numpy.zeros((n,300*200),dtype=int)

"""
train_images = file_open("./train/c0/","*.jpg")
print "Train image : "+str(train_images)
n = len(train_images)
print "Number of images : "+str(n)
"""

#filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./train/c0/*.jpg"))
#print "Queue value : "+str(filename_queue)

"""
for i in range(n):
    train_images[i] = flatten_image(matrix_image(train_images[i]))
"""
