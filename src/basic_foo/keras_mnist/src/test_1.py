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
#from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, UpSampling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint



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


# network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45

def net_1(x_train, y_train,x_test, y_test):
  """
  Create the MLP here - Multi Layer Perceptron 
  Compile it below in the - net_1_compile()
  """
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Activation, Dropout
  from tensorflow.keras.utils import to_categorical, plot_model
  # compute the number of labels
  num_labels = len(np.unique(y_train))
  print("----num_labels---",num_labels)
  # OneHot Encoding 
  y_train = to_categorical(y_train)
  print("-np.shape(y_train)--",np.shape(y_train))
  y_test = to_categorical(y_test)
  image_size = x_train.shape[1] # image dimensions (assumed square)
  print("----image_size----image_height,image_width-",image_size)
  # NOT IMG SIZE -- But Measure of one SIDE of SQUARE - image_height,image_width ? 
  input_size = image_size * image_size
  print("----input_size---",input_size)
  # resize and normalize
  x_train = np.reshape(x_train, [-1, input_size])
  print("--np.shape(x_train)-aa-",np.shape(x_train))
  x_train = x_train.astype('float32') / 255
  print("---np.shape(x_train)-bb-",np.shape(x_train))
  x_test = np.reshape(x_test, [-1, input_size])
  x_test = x_test.astype('float32') / 255

  #
  # model is a 3-layer MLP with ReLU and dropout after each layer
  model = Sequential()
  model.add(Dense(hidden_units, input_dim=input_size))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Dense(hidden_units))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Dense(num_labels))
  # this is the output for one-hot vector
  model.add(Activation('softmax'))
  print(model.summary()) #
  for layer in model.layers:
    print("===== net_1_mlp_mnist ======")
    print("--layer.name-",layer.name)
    print("--layer.inbound_nodes-",layer.inbound_nodes)
    print("--layer.outbound_nodes-",layer.outbound_nodes)
    # print(layer.name, layer.inbound_nodes, layer.outbound_nodes)
  plot_model(model, to_file='net_1_mlp_mnist.png', show_shapes=True)
  # TODO -- Diff First Layer as shown in the -- net_1_mlp_mnist.png 
  return model

def net_1_compile(model,x_train, y_train,x_test, y_test):
  """
  compile and fit the Model 
  # loss function for one-hot vector
  # use of adam optimizer
  # accuracy is good metric for classification tasks
  """
  model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])
  # train the network
  model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
  return model

def net_1_validate(model,x_train, y_train,x_test, y_test):
  """
  """
  # validate the model on test dataset to determine generalization
  _, acc = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
  print("\nTest accuracy: %.1f%%" % (100.0 * acc))

def adv_auto_enc():
  """
  Adversarial Auto Encoder
  # count the number of unique train labels
  """
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  unique_train, counts_train = np.unique(y_train, return_counts=True) #print("--type(unique)--",type(unique_train)) # ndArray
  
  print("Unique TRAIN DATA Labels and IMAGE Counts: ", dict(zip(unique_train, counts_train)))
  #{0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
  unique_test, counts_test = np.unique(y_test, return_counts=True)
  print("Unique TEST DATA Labels and IMAGE Counts: ", dict(zip(unique_test, counts_test)))
  #{0: 980, 1: 1135, 2: 1032, 3: 1010, 4: 982, 5: 892, 6: 958, 7: 1028, 8: 974, 9: 1009}
  print("Shape--> x_train , y_train--, x_test, y_test--->",x_train.shape[0],y_train.shape[0],x_test.shape[0],y_test.shape[0])
  idx_25 = np.random.randint(0, x_train.shape[0], size=25) # sample 25 mnist digits from train dataset
  images = x_train[idx_25] #print("--type(images)--",type(images)) # ndArray
  labels = y_train[idx_25] #print("--type(labels)--",type(labels)) # ndArray
  img_1 = images[0] #get first image from Random SAMPLE of 25 Images 
  print("---",img_1.shape[0]) #WIDTH -  image_width
  print("---",img_1.shape[1]) #HEIGHT - image_height
  
  fig, axes = plt.subplots(2,10, figsize = (16, 4))

  count = 0

  for i in range(2):
      for j in range(10):
          axes[i,j].imshow(x_train[count], cmap = 'gray')
          count+=1
  # Normalize data 
  x_train = x_train / 255.0
  x_test = x_test / 255.0
  #
  # Add Noise to the INIT MNIST DATA and prepare NOISY source data
  noise_factor = 0.1

  x_train_noise = x_train + noise_factor * np.random.normal(loc = 0., scale = 1., size = x_train.shape)
  x_test_noise = x_test + noise_factor * np.random.normal(loc = 0., scale = 1., size = x_test.shape)
  #
  print("---x_train.shape---",x_train.shape)
  #
  fig, axes = plt.subplots(2,10, figsize = (16,4))
  count = 0
  for i in range(2):
      for j in range(10):
          axes[i,j].imshow(x_train_noise[count], cmap = 'gray')
          count+=1
  #
  x_train = x_train.reshape(x_train.shape[0], 28 ,28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28,28,1)

  x_train_noise = x_train_noise.reshape(x_train_noise.shape[0], 28 ,28, 1)
  x_test_noise = x_test_noise.reshape(x_test_noise.shape[0], 28,28,1)

  print("---x_train.shape, x_train_noise.shape--->",x_train.shape, x_train_noise.shape)
  #Done above in --> def get_init_config()
  # devices = tf.config.experimental.list_physical_devices("GPU")
  # tf.config.experimental.set_memory_growth(devices[0], enable = True)
  #
  # encoder
  encoder_input = Input(shape = x_train.shape[1:])
  x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(encoder_input)
  x = BatchNormalization()(x)
  x = MaxPool2D(pool_size = (2,2), padding = 'same')(x)
  x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(x)
  x = BatchNormalization()(x)
  encoded = MaxPool2D(pool_size = (2,2), padding = 'same')(x)

  ## decoder
  x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(encoded)
  x = BatchNormalization()(x)
  x = UpSampling2D()(x)
  x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(x)
  x = BatchNormalization()(x)
  x = UpSampling2D()(x)
  decoded = Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same')(x)

  #
  autoencoder = Model(encoder_input, decoded, name = 'Denoising_autoencoder')
  #autoencoder.summary()
  print(autoencoder.summary()) #
  for layer in autoencoder.layers:
    print("===== autoencoder ======")
    print("--layer.name-",layer.name)
    print("--layer.inbound_nodes-",layer.inbound_nodes)
    print("--layer.outbound_nodes-",layer.outbound_nodes)
    # print(layer.name, layer.inbound_nodes, layer.outbound_nodes)
  tf.keras.utils.plot_model(autoencoder, to_file="model_autoencoder_1.png", show_shapes=True)
  
  # Compile Model 
  autoencoder.compile(loss = 'binary_crossentropy', optimizer = 'adam')
  # Train Model 
  checkpoint = ModelCheckpoint("denoising_model.h5", save_best_only=True, save_weights_only=False, verbose = 1)
  history = autoencoder.fit(x_train_noise, x_train, batch_size = 5, epochs = 5, callbacks = checkpoint, validation_split = 0.25, verbose = 2)
  ## ptx Issues as documented in the TerMinal Log Files 
  """
  08_23/kera_autoEnc/src/term_log_autoencoder_1.log
  
  2023-09-12 22:07:26.524036: F tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc:492] ptxas returned an error during compilation of ptx to sass: 'INTERNAL: ptxas exited with non-zero error code 65280, output: ptxas /tmp/tempfile-dhankar-1-74bb1998-9107-6052c0edf8252, line 5; fatal   : Unsupported .version 7.1; current version is '7.0'
  ptxas fatal   : Ptx assembly aborted due to errors
  '  If the error message indicates that a file could not be written, please verify that sufficient filesystem space is provided.

  """

  
if __name__ == "__main__":
  #
  get_init_config()
  # x_train, y_train,x_test, y_test = get_mnist() #mnist_data = 
  # model_net_1_mlp = net_1(x_train, y_train,x_test, y_test)
  # fit_model_net_1_mlp = net_1_compile(model_net_1_mlp,x_train, y_train,x_test, y_test)
  #net_1_validate(fit_model_net_1_mlp,x_train, y_train,x_test, y_test)

  # model = create_model_seq(mnist_data) #input_shape = (None, 32, 32, 3)
  # model.build() #model.build(input_shape)
  # get_summary(model)
  adv_auto_enc()

    

