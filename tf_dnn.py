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

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


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

def image_process(row):
    image = Image.open(image_path)
    image = image.resize((320,240))
    train_images.append(np.array(image))
    #print image

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
    #print image

def preprocess():
    data_df = pandas.read_csv('./driver_imgs_list.csv',na_filter=False)
    print(data_df.columns)
    print(data_df.shape)
    trainSize = int(data_df.shape[0]*0.8)
    testSize = data_df.shape[0] - trainSize

    np_data = data_df.as_matrix()
    print ("Data shape : "+str(np_data.shape))

    aperm = np.random.permutation(data_df.shape[0])
    train_data = np_data[aperm[0:trainSize],:]
    test_data = np_data[aperm[trainSize:],:]

    print ("Train data shape : "+str(train_data.shape))
    print ("Test data shape : "+str(test_data.shape))

    for row in train_data:
        get_file_names(row,train_path_list,train_labels)


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

#print "Train path list : "+str(train_path_list)
#print "Train labels : "+str(train_labels)

# csv_path = tf.train.string_input_producer(['./data/driver_imgs_list.csv'])
# textReader = tf.TextLineReader()
# _, csv_content = textReader.read(csv_path)
# subj, class_name, image = tf.decode_csv(csv_content, record_defaults=[[""], [""], [""]])



preprocess()

"""
Building a queue for sending the data
"""
batch_size = 100
images = ops.convert_to_tensor(train_path_list, dtype=dtypes.string)
#labels = ops.convert_to_tensor(train_labels, dtype=dtypes.ndarr)
labels = tf.constant(train_labels)
filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

reader = tf.read_file(filename_queue[0])
image = tf.image.decode_jpeg(reader, channels=1)
image.set_shape([480, 640, 1])
image = tf.cast(image,tf.float64)
labels = filename_queue[1]
images_batch, labels_batch = tf.train.batch([image, labels], batch_size=batch_size, capacity=batch_size * 2,num_threads=10)


"""
Model
"""
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()


#image augmentation
#img_aug = ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_rotation(max_angle=25.)
#img_aug.add_random_blur(sigma_max=3.)

def get_keras_cnn_model():


    model = Sequential()
    # model.add(Input(shape=(480, 640, 1),name="input"))

    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv1',input_shape=(480,640,1)))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv3'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv4'))
    model.add(ELU())

    model.add(Flatten())
    model.add(Dense(512, activation='softmax', name='output1'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', name='output2'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

checkpointer = ModelCheckpoint(filepath="./keras_checkpoint/weights.hdf5", verbose=1, save_best_only=True)

with tf.Session() as sess:
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    model = get_keras_cnn_model()
    datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
    # Get an image tensor and print its value.
    #for i in range(len(train_labels)/batch_size):
    for i in range(int(len(train_path_list)/batch_size+1)):
        #current_img = sess.run([image_tensor])
        #print "Image tensor : "+str(current_img[0].shape)
        #print "Label : "+str(train_labels[i])
        xs,ys = sess.run([images_batch,labels_batch])
        #xs = sess.run(images_batch)
        print ("Before "+ str(xs.shape))
        xs = xs.reshape((batch_size,480,640,1))
        print (ys.shape)
        val_xs = xs[:10]
        val_ys = ys[:10]
	#print val_xs.shape, val_ys.shape
        #model.fit({'input':xs}, ys,validation_set=(val_xs,val_ys), n_epoch=10, run_id="model",show_metric=True)
        datagen.fit(xs)
        hist= model.fit_generator(datagen.flow(xs, ys), samples_per_epoch=batch_size,validation_data=(val_xs,val_ys),callbacks=[checkpointer],nb_epoch=10)
        print(hist.history)
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    model.save("cnn_statefarm")

#print filename_queue.size()

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

