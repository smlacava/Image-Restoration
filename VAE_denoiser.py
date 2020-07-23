from tensorflow.keras.layers import Lambda, Input, Dense, Layer, Conv2D, BatchNormalization, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from image_denoiser import image_denoiser

class VAE_denoiser(image_denoiser):
    def __init__(self, name = 'Image_Denoiser'):
        """
        The __init__ method is the initializer
        :param name: it is the name of the autoencoder ('Image_Denoiser' by default), and it can be useful in saving the
                     weights of the trained encoder (in this case, the path has to be included in the neme)
        """
        tf.compat.v1.disable_eager_execution()
        self.latent_dim = 2
        self.filter_dim = 32
        self.kernel_dim = 3
        self.name = name
        self.image_dimension = [28, 28, 1]
        self.autoencoder = self.autoencoder_creation()
        self.patience = 10

    def autoencoder_creation(self):
      """
      The autoencoder_creation method creates the CNN structure which will be used as variational autoencoder.
      :return: autoencoder: it is the generated variational autoencoder
      """
      i       = Input(shape=self.image_dimension, name='encoder_input')
      cx      = Conv2D(filters=self.filter_dim, kernel_size=self.kernel_dim, strides=2, padding='same', activation='relu')(i)
      cx      = BatchNormalization()(cx)
      cx      = Conv2D(filters=self.filter_dim*2, kernel_size=self.kernel_dim, strides=2, padding='same', activation='relu')(cx)
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
      cx    = Conv2DTranspose(filters=self.filter_dim*2, kernel_size=self.kernel_dim, strides=2, padding='same', activation='relu')(x)
      cx    = BatchNormalization()(cx)
      cx    = Conv2DTranspose(filters=self.filter_dim, kernel_size=self.kernel_dim, strides=2, padding='same',  activation='relu')(cx)
      cx    = BatchNormalization()(cx)
      o     = Conv2DTranspose(filters=self.image_dimension[-1], kernel_size=self.kernel_dim, activation='sigmoid', padding='same', name='decoder_output')(cx)

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
          reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * self.image_dimension[0] * self.image_dimension[1] * self.image_dimension[2]
          kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
          kl_loss = K.sum(kl_loss, axis=-1)
          kl_loss *= -0.5
          return K.mean(reconstruction_loss + kl_loss)
      
      autoencoder.compile(optimizer='adam', loss=vae_loss)
      return autoencoder

    def predict(self, test_noisy, test = None, n = None, return_loss = False):
        """
        The predict method denoises a set of images, and compares them with the corresponding images without noise (if
        they are used as argument).
        :param test_noisy: it is the set of noised images
        :param test: it is the set of the images without noise, corresponding to the noised images (None by default)
        :param n: it is the number of imaged which have to be compared (None by default)
        :param return_loss: it has to be True if the function has to return also the reconstruction loss (False by default)
        :return: the denoised images, and the binary crossentropy between each (noised) image and its reconstruction (if return_loss is True)
        """
        test_check = not (test is None)
        if test_check:
            test = self._preprocessing(test)
        test_noisy = self._preprocessing(test_noisy)
        decoded_imgs = self.autoencoder.predict(test_noisy)
        if self.image_dimension[2] == 1:
            decoded_imgs = decoded_imgs.reshape(len(decoded_imgs), self.image_dimension[0], self.image_dimension[1])
        if not (n is None):
            self._plot_results(n, test_noisy, test, decoded_imgs)

        if return_loss and test_check:
           return decoded_imgs, self._compute_loss(test, decoded_imgs)
        return decoded_imgs

    def _compute_loss(self, test_noisy, decoded_imgs):
        """
        The method compute_loss compotes the reconstruction loss between the test images (noised) and the denoised ones
        :param test_noisy: it is the dataset of noised images used as test set
        :param decoded_imgs: it is the dataset of denoised images
        :return: the binary crossentropy for each image of the dataset
        """
        if self.image_dimension[2] == 1:
            test_noisy = test_noisy.reshape(len(test_noisy), self.image_dimension[0], self.image_dimension[1])
        n_samples = len(test_noisy)
        reconstruction_loss = np.zeros(shape=(n_samples, ))
        n_elements = np.prod(self.image_dimension)
        for i in range(n_samples):
          x = np.reshape(test_noisy[i], n_elements)
          y = np.clip(np.reshape(decoded_imgs[i], n_elements), 1e-12, 1. - 1e-12)
          reconstruction_loss[i] = -np.sum(x*np.log(y+1e-9))/n_elements
        return reconstruction_loss


if __name__ == "__main__":
    print("The VAE_denoiser class allows to create variational autoencoders which filter out noise from input images.")
    print("In particular, the variational autoencoder object is a CNN, which takes 28 x 28 images (otherwise, it ", 
          "automatically resamples them)")
    print("\nTo make an example, it is possible to use the mnist dataset:")
    print("\n     from keras.datasets import mnist")
    print("     (x_train, _), (x_test, _) = mnist.load_data()")
    print("\nNow, it is possible to add some noise to the images:")
    print("\n     noise_factor = 64")
    print("     x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)")
    print("     x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)")
    print("\nIt is needed to limit the range of the values between 0 and 255:")
    print("\n     x_train_noisy = np.clip(x_train_noisy, 0., 1.)")
    print("     x_test_noisy = np.clip(x_test_noisy, 0., 1.)")
    print("\nFinally, it is possible to initialize the denoiser:")
    print("\n     ID = VAE_denoiser()")
    print("\nThere are different ways to train it:")
    print("\t - It is possible to avoid the validation, thruogh ID.fit(x_train_noisy, x_train)")
    print("\t - It is possible to use as validation set a fraction of the training set, through ID.fit(x_train_noisy, ",
          "x_train, n), where n is a value among 0 and 1")
    print("\t - It is possible to insert a validation set and the corresponing noised one, through ",
          "ID.fit(x_train_noisy, x_train, x_val_noisy, x_val), where x_val and x_val_noisy are the validation set ",
          "and the noised one")
    print("Furthermore, it is possible to set the number of epochs and the size of the batch.")
    print("For example, in order to train it in 100 epochs with the previously computed training set and the noised ",
          "one, using the +20% of them as validation set, and with a batch size equal to 128, it is possible to use:")
    print("\n     ID.fit(x_train_noisy, x_train, 0.2, epochs=100, batch_size=128)")
    print("\nIn order to denoise the test set and to see the resulting images, for example the first ten images, it is",
          "possible to use:")
    print("\n     ID.predict(x_test_noisy, x_test, 10)")
