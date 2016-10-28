import tensorflow as tf
import numpy as np
import glob
import os
import pandas
from PIL import Image

global_image_dir = "./data/train/"
file_open = lambda x,y: glob.glob(os.path.join(x,y))
train_path_list = []
test_path_list = []

def get_file_names(row, path):
    image_path = global_image_dir+row[1]+"/"+row[2]
    path.append(image_path)
    image = Image.open(image_path)
    image = image.resize((320,240))
    #train_images.append(np.array(image))
    print image

def image_process(row):
    image_path = global_image_dir+row[1]+"/"+row[2]
    print image_path
    image_queue = tf.train.string_input_producer([image_path])
    rteader = tf.WholeFileReader()
    key, value = reader.read(image_queue)
    current_img = tf.image.decode_jpeg(value)
    print current_img
    matrix_image(current_img)

def matrix_image(image):
    image = list(image.getdata())
    image = Image.open(image)
    image = map(list,image)
    image = np.array(image)
    print image


data_df = pandas.read_csv('./data/driver_imgs_list.csv')
print(data_df.columns)
colImg = data_df['img']

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
    get_file_names(row,train_path_list)

print "Train path list : "+str(train_path_list)

filename_queue = tf.train.string_input_producer(train_path_list)
image_reader = tf.WholeFileReader()




print filename_queue.size

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
