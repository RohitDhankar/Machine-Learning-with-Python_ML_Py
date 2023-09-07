#conda activate env_tf2

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


def get_init_config():
  """
  """
  #tf.config.list_physical_devices('GPU')
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  import nvidia.cudnn
  print(nvidia.cudnn.__file__) #/home/dhankar/anaconda3/envs/env_tf2/lib/python3.9/site-packages/nvidia/cudnn/__init__.py



batch_size = 128
num_classes = 10
epochs = 2
image_height, image_width = 28, 28


def calc_params():
  """
  params_calc
  """
  count_of_neurons_l1 = bias_vals_l1 = 12
  input_dim_l1 = 8
  count_of_neurons_l2 = bias_vals_l2 = 32
  input_dim_l2 = 8
  
  r_u = 'random_uniform'
  r_n = "random_normal"

  params_calc_l1 = input_dim_l1 * count_of_neurons_l1 + bias_vals_l1
  print("--params_calc_l1--->",params_calc_l1)
  params_calc_l2 = input_dim_l2 * count_of_neurons_l2 + bias_vals_l2
  print("--params_calc_l2--->",params_calc_l2)

  return r_n , r_u

def get_mnist():
  """
  #Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
  """
  mnist = tf.keras.datasets.mnist # print("--type(mnist---",type(mnist)) #<class 'module'>
  (x_train, y_train),(x_test, y_test) = mnist.load_data()
  # count the number of unique train labels
  unique, counts = np.unique(y_train, return_counts=True)
  print("--type(unique)--",type(unique)) # ndArray
  print("unique TRAIN DATA labels: ", dict(zip(unique, counts)))

  unique, counts = np.unique(y_test, return_counts=True)
  print("unique TEST DATA labels: ", dict(zip(unique, counts)))
  #
  print("--shape of the x_train and y_train--->",x_train.shape[0],y_train.shape[0])
  # sample 25 mnist digits from train dataset
  indexes = np.random.randint(0, x_train.shape[0], size=25)
  images = x_train[indexes]
  print("--type(images)--",type(images)) # ndArray
  labels = y_train[indexes]
  print("--type(labels)--",type(labels)) # ndArray
  # plot the 25 mnist digits
  # plt.figure(figsize=(5,5))
  # for i in range(len(indexes)):
  #   plt.subplot(5, 5, i + 1)
  #   image = images[i]
  #   plt.imshow(image, cmap='gray')
  #   plt.axis('off')
  #   plt.savefig("mnist-samples.png")
  #   plt.show()
  #   plt.close('all')

  #print("---type(x_train---",type(x_train)) #--type(x_train--- <class 'numpy.ndarray'>
  print("---type(x_test---np.shape(x_test)-",np.shape(x_test)) #(10000, 28, 28)
  print("---type(x_test---np.shape(x_train)-",np.shape(x_train)) #(60000, 28, 28)
  # x_train, x_test = x_train / 255.0, x_test / 255.0
  # print("---type(x_test---np.shape(x_test)-",np.shape(x_test))
  # print("---type(x_test---np.shape(x_train)-",np.shape(x_train))

  # x_train = x_train.reshape(x_train.shape[0], image_height,image_width,1)
  # x_test = x_test.reshape(x_test.shape[0], image_height, image_width,1)
  # print("---type(x_test---np.shape(x_test)-",np.shape(x_test)) #(10000, 28, 28, 1)
  # print("---type(x_test---np.shape(x_train)-",np.shape(x_train))

  # Numpy slices - Head and Tail 
  # print("---x_test[:2]---",x_test[:2])
  # print("---x_test[-2:]---",x_test[-2:])
  # print("---x_train[:2]---",x_train[:2])
  # print("---x_train[-2:]---",x_train[-2:])

  img_1 = x_test[1]
  print("--type(img_1)-",type(img_1))
  print("--np.shape(img_1)---",np.shape(img_1))

  # cv2.imshow('First sample', x_train[0])
  # cv2.waitKey(5000)
  # cv2.destroyWindow('First sample')
  # cv2.imshow("image", img_1)
  # cv2.waitKey()

  
  input_shape = (image_height, image_width,1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  # x_train /= 255
  # x_test /= 255
  # y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  # y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  
  # mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
  # print("--type(mnist---",type(mnist_data)) 

  return x_train, y_train,x_test, y_test #mnist_data , input_shape

def create_model_seq(mnist_data):
  """
  """
  model = Sequential() #print(model) ##<tensorflow.python.keras.engine.sequential.Sequential object at 0x7fbe54143100>
  model.add(Dense(count_of_neurons_l1,input_dim=input_dim_l1, kernel_initializer=r_u))
  model.add(Dense(count_of_neurons_l2,input_dim=input_dim_l2, kernel_initializer=r_u))
  # model.add(Dense(units=64, activation='relu'))
  # model.add(Dense(units=10, activation='softmax'))

  model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.SGD(
                    learning_rate=0.01, momentum=0.9, nesterov=True))

  return model


def get_summary(model):
  """
  """
  print(model.summary()) #
  for layer in model.layers:
    print("===========")
    print("--layer.name-",layer.name)
    print("--layer.inbound_nodes-",layer.inbound_nodes)
    print("--layer.outbound_nodes-",layer.outbound_nodes)
    # print(layer.name, layer.inbound_nodes, layer.outbound_nodes)
  tf.keras.utils.plot_model(model, to_file="model_2.png", show_shapes=True)


def net_1(x_train, y_train,x_test, y_test):
  """
  """
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Activation, Dropout
  from tensorflow.keras.utils import to_categorical, plot_model
  # compute the number of labels
  num_labels = len(np.unique(y_train))
  print("----num_labels---",num_labels)
  # OneHot Encoding 
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  

if __name__ == "__main__":
  #
  get_init_config()
  x_train, y_train,x_test, y_test = get_mnist() #mnist_data = 
  net_1(x_train, y_train,x_test, y_test)
  # model = create_model_seq(mnist_data) #input_shape = (None, 32, 32, 3)
  # model.build() #model.build(input_shape)
  # get_summary(model)

    

