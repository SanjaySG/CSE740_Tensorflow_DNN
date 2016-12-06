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
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import math

global_image_dir = "./train/"
file_open = lambda x,y: glob.glob(os.path.join(x,y))
train_path_list = []
train_labels = []
test_path_list = []
test_labels = []
train_images = []

def get_file_names(row, path,label):
    image_path = global_image_dir+row[1]+"/"+row[2]
    path.append(image_path)
    #temp_label = np.zeros((10))
    temp_label = [0 for x in range(10)]
    temp_label[int(row[1][1])] = 1
    label.append(temp_label)

"""
def image_process(row):
    image = Image.open(image_path)
    image = image.resize((320,240))
    #train_images.append(np.array(image))
    print(image)

def matrix_image(image):
    image = list(image.getdata())
    image = Image.open(image)
    image = map(list,image)
    image = np.array(image)
    print(image)
"""

def preprocess():
    data_df = pandas.read_csv('./driver_imgs_list.csv',na_filter=False)
    print(data_df.columns)
    print(data_df.shape)
    trainSize = int(data_df.shape[0]*0.8)
    testSize = data_df.shape[0] - trainSize

    np_data = data_df.as_matrix()
    print("Data shape : "+str(np_data.shape))

    aperm = np.random.permutation(data_df.shape[0])
    train_data = np_data[aperm[0:trainSize],:]
    test_data = np_data[aperm[trainSize:],:]

    print ("Train data shape : "+str(train_data.shape))
    print ("Test data shape : "+str(test_data.shape))

    for row in train_data:
        get_file_names(row,train_path_list,train_labels)

    for row in test_data:
        get_file_names(row,test_path_list,test_labels)

"""
Model
"""

def get_cnn_model():
    network = input_data(shape=[None, 480, 640, 1],name='input',data_preprocessing=img_prep)

    # Step 1: Convolution
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 2: Max pooling
    network = max_pool_2d(network, 2)

    # Step 3: Convolution
    network = conv_2d(network, 64, 3, activation='relu')

    # Setp 4: mac pool
    network = max_pool_2d(network, 2)

    # Step 5: Convolution
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 6: Max pooling
    network = max_pool_2d(network, 2)

    # Step 7: Convolution
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 8: Max pooling
    network = max_pool_2d(network, 2)

    # Step 9: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')

    # Step 10: Dropout - to prevent over-fitting
    network = dropout(network, 0.5)

    # Step 11: Fully-connected neural network with 10 outputss
    network = fully_connected(network, 10, activation='softmax')

    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)

    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="checkpoint/")
    #
    return model


def get_auto_enc():
    network = tflearn.input_data(shape=[None, 307200])

    encoder1 = tflearn.fully_connected(network, 512)

    encoder2 = tflearn.fully_connected(encoder1, 256)

    decoder2 = tflearn.fully_connected(encoder2, 256)

    decoder1 = tflearn.fully_connected(delcoder2, 512)

    output = tflearn.fully_connected(decoder1, 307200)

    # Regression, with mean square error
    net = tflearn.regression(output, optimizer='adam', learning_rate=0.001,loss='mean_square', metric=None)

    # Training the auto encoder
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model

preprocess()

"""
Building a queue for sending the data
"""
batch_size = 10

def convert_to_batch(path_list,label_list):
    images = ops.convert_to_tensor(path_list, dtype=dtypes.string)
    labels = tf.constant(label_list)
    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

    reader = tf.read_file(filename_queue[0])
    image = tf.image.decode_jpeg(reader, channels=1)
    image.set_shape([480, 640, 1])
    image = tf.cast(image,tf.float64)
    labels = filename_queue[1]
    images_batch, labels_batch = tf.train.batch([image, labels], batch_size=batch_size, capacity=batch_size * 2,num_threads=10)

    return images_batch, labels_batch

"""
Session run
"""
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

#image augmentation
#img_aug = ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_rotation(max_angle=25.)

with tf.Session() as sess:
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    model = get_cnn_model()

    images_batch, labels_batch = convert_to_batch(train_path_list,train_labels)
    # Get an image tensor and print its value.
    #for i in range(len(train_labels)/batch_size):
    for i in range(int(len(train_path_list)/batch_size)):
        xs,ys = sess.run([images_batch,labels_batch])
        print("Before "+ str(xs.shape))
        xs = xs.reshape((batch_size,480,640,1))
        print(ys.shape)
        val_xs = xs[:10]
        val_ys = ys[:10]
        #print val_xs.shape, val_ys.shape
        model.fit({'input':xs}, ys,validation_set=(val_xs,val_ys), n_epoch=10, run_id="model",show_metric=True)

    print("Training complete")
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    model.save("cnn_statefarm")
    print("Model saved")

    print("Ruuning evalute on test data")
    images_batch,labels_batch = convert_to_batch(test_path_list,test_labels)
    for i in range(int(len(train_path_list)/batch_size)):
        testX,testY = sess.run([images_batch,labels_batch])
        testX = testX.reshape((batch_size,480,640,1))
        model.evaluate(testX,testY,batch_size)

    print("Testing complete")
