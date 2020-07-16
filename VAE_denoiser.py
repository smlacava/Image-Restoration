import keras
from keras.layers import Lambda, Input, Dense, Layer, Conv2D, BatchNormalization, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
#from image_denoiser import image_denoiser


class VAE_denoiser(image_denoiser):
    def __init__(self, name = 'Image_Denoiser'):
        """
        The __init__ method is the initializer
        :param name: it is the name of the autoencoder ('Image_Denoiser' by default), and it can be useful in saving the
                     weights of the trained encoder (in this case, the path has to be included in the neme)
        """
        self.latent_dim = 2
        self.filter_dim = 8
        self.name = name
        self.image_dimension = [28, 28, 1]
        self.autoencoder = self.autoencoder_creation()

    def autoencoder_creation(self):
      i       = Input(shape=self.image_dimension, name='encoder_input')
      cx      = Conv2D(filters=self.filter_dim, kernel_size=3, strides=2, padding='same', activation='relu')(i)
      cx      = BatchNormalization()(cx)
      cx      = Conv2D(filters=self.filter_dim*2, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
      cx      = BatchNormalization()(cx)
      x       = Flatten()(cx)
      x       = Dense(20, activation='relu')(x)
      x       = BatchNormalization()(x)
      mu      = Dense(self.latent_dim, name='latent_mu')(x)
      sigma   = Dense(self.latent_dim, name='latent_sigma')(x)

      conv_shape = K.int_shape(cx)

      def sample_z(args):
          """
          The sample_z function samples the value for z from the computed μ and σ values by resampling into 
          μ + K.exp(σ/2) * ϵ (the reparameterization allows to obtain a differentiable loss function)
          :param name: it is the composed by μ and σ values
          :return: the reparametrized function value
          """
          mu, sigma = args
          batch     = K.shape(mu)[0]
          dim       = K.int_shape(mu)[1]
          eps       = K.random_normal(shape=(batch, dim))
          return mu + K.exp(sigma / 2) * eps

      z       = Lambda(sample_z, output_shape=(self.latent_dim, ), name='z')([mu, sigma])
      encoder = Model(i, [mu, sigma, z], name='encoder')

      d_i   = Input(shape=(self.latent_dim, ), name='decoder_input')
      x     = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
      x     = BatchNormalization()(x)
      x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
      cx    = Conv2DTranspose(self.filter_dim*2, kernel_size=3, strides=2, padding='same', activation='relu')(x)
      cx    = BatchNormalization()(cx)
      cx    = Conv2DTranspose(filters=self.filter_dim, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
      cx    = BatchNormalization()(cx)
      o     = Conv2DTranspose(filters=self.image_dimension[-1], kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)

      decoder = Model(d_i, o, name='decoder')

      autoencoder_outputs = decoder(encoder(i)[2])
      autoencoder         = Model(i, autoencoder_outputs, name='vae')

      def vae_loss(true, pred):
          """
          The vae_loss function defines an optimized estimate of the loss value, considering the weighted binary 
          cross-entropy (as the reconstruction loss) and the Kullback-Leibler divergence loss (KL loss).
          :parameter true: it is the target image
          :parameter pred: it is the predicted image
          :return: the average between the reconstruction loss and the KL loss
          """
          reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * self.image_dimension[0] * self.image_dimension[1]
          kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
          kl_loss = K.sum(kl_loss, axis=-1)
          kl_loss *= -0.5
          return K.mean(reconstruction_loss + kl_loss)
      
      autoencoder.compile(optimizer='adam', loss=vae_loss)
      return autoencoder
  
    def predict(self, test_noisy, test=None, n=None):
        """
        The predict method denoises a set of images, and compares them with the corresponding images without noise (if
        they are used as argument).
        :param test_noisy: it is the set of noised images
        :param test: it is the set of the images without noise, corresponding to the noised images (None by default)
        :param n: it is the number of imaged which have to be compared
        :return: the denoised images
        """
        if not (test is None):
            test = self._preprocessing(test)
        test_noisy = self._preprocessing(test_noisy)
        decoded_imgs = self.autoencoder.predict(test_noisy)
        if self.image_dimension[2] == 1:
            decoded_imgs = decoded_imgs.reshape(len(decoded_imgs), self.image_dimension[0], self.image_dimension[1])
        if not (n is None):
            self._plot_results(n, test_noisy, test, decoded_imgs)
        return decoded_imgs

    def fit(self, train_noisy, train, val_noisy=None, val=None, epochs=100, batch_size=None):
        """
        The fit method is used to train the CNN, and it shows the loss curve(s) at the end of the whole process.
        :param train_noisy: it is the dataset of the noised images which is used in the training phase
        :param train: it is the dataset of images, without noise, which is used in the training phase
        :param val_noisy: it can be the dataset of noised images which is used in the validation phase, or a number
                           between 0 and 1 which indicates the fraction of training dataset which has to be used as
                           validation set, or None to avoid the validation phase (None by default)
        :param val: it is the dataset of images, without noise, which is used in the validation phase (None by default)
        :param epochs: it is the number of fitting epochs, which is automatically reduced if the validation loss does
                       not reduces anymore after a certain number of steps, and in any case the settings which shows the
                       best performance are saved (100 by default)
        :param batch_size: it defines the number of samples that will be propagated through the network (the minimum
                           between 128 and the total number of training samples by default)
        """
        self.set_input_dimensions(train.shape[1:])
        
        train = self._preprocessing(train)
        train_noisy = self._preprocessing(train_noisy)
        if batch_size is None:
            batch_size = min(128, len(train))
            print('Batch size:' + str(batch_size))
        if val_noisy is None and val is None:
            check = 0
            history = self.autoencoder.fit(train_noisy, train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           callbacks=[
                                               TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
        elif isinstance(val_noisy, float):
            if batch_size is None:
                batch_size = min(128, int(len(train) * (1 - val_noisy)))
                print('Batch size:' + str(batch_size))
            check = 1
            history = self.autoencoder.fit(train_noisy, train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           validation_split=val_noisy,
                                           shuffle=True,
                                           callbacks=[
                                               TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False),
                                               EarlyStopping(monitor='val_loss', patience=3),
                                               ModelCheckpoint('/tmp/checkpoint', monitor='val_loss', mode='min',
                                                               verbose=1, save_best_only=True, save_weights_only=True)])
        else:
            check = 1
            val = self._preprocessing(val)
            val_noisy = self._preprocessing(val_noisy)
            history = self.autoencoder.fit(train_noisy, train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           validation_data=(val_noisy, val),
                                           callbacks=[
                                               TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False),
                                               EarlyStopping(monitor='val_loss', patience=3),
                                               ModelCheckpoint('/tmp/checkpoint', monitor='val_loss', mode='min',
                                                               verbose=1, save_best_only=True, save_weights_only=True)])
        if check == 1:
            self.autoencoder.load_weights('/tmp/checkpoint')
            plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'],
                     range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
            plt.title('Loss Curves')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Model', 'Validation'])
            plt.show()
        else:
            plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()
