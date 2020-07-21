from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class image_denoiser():
    def __init__(self, name='Image_Denoiser'):
        """
        The __init__ method is the initializer
        :param name: it is the name of the autoencoder ('Image_Denoiser' by default), and it can be useful in saving the
                     weights of the trained encoder (in this case, the path has to be included in the neme)
        """
        self.name = name
        self.filter_dim = 32
        self.kernel_dim = 3
        self.image_dimension = [28, 28, 1]
        self.patience = 3
        self.autoencoder = self.autoencoder_creation()

    def autoencoder_creation(self):
        """
        The autoencoder_creation method creates the CNN structure which will be used as autoencoder.
        :return: autoencoder: it is the generated autoencoder
        """
        input_img   = Input(shape=(self.image_dimension[0], self.image_dimension[1], self.image_dimension[2]))
        x           = Conv2D(filters=self.filter_dim, kernel_size=self.kernel_dim, activation='relu', padding='same')(input_img)
        x           = MaxPooling2D((2, 2), padding='same')(x)
        x           = Conv2D(filters=self.filter_dim, kernel_size=self.kernel_dim, activation='relu', padding='same')(x)
        encoded     = MaxPooling2D((2, 2), padding='same')(x)
        x           = Conv2D(filters=self.filter_dim, kernel_size=self.kernel_dim, activation='relu', padding='same')(encoded)
        x           = UpSampling2D((2, 2))(x)
        x           = Conv2D(filters=self.filter_dim, kernel_size=self.kernel_dim, activation='relu', padding='same')(x)
        x           = UpSampling2D((2, 2))(x)
        decoded     = Conv2D(filters=self.image_dimension[2], kernel_size=self.kernel_dim, activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        return autoencoder

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
                                               EarlyStopping(monitor='val_loss', patience=self.patience),
                                               ModelCheckpoint('/tmp/checkpoint', monitor='val_loss', mode='min',
                                                               verbose=1, save_best_only=True, 
                                                               save_weights_only=True)])
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
                                               EarlyStopping(monitor='val_loss', patience=self.patience),
                                               ModelCheckpoint('/tmp/checkpoint', monitor='val_loss', mode='min',
                                                               verbose=1, save_best_only=True, 
                                                               save_weights_only=True)])
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

    def predict(self, test_noisy, test=None, n=None, return_loss = False):
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
        bce = tf.keras.losses.BinaryCrossentropy()
        n_samples = len(test_noisy)
        reconstruction_loss = np.zeros(shape=(n_samples, ))
        for i in range(n_samples):
          reconstruction_loss[i] = bce(test_noisy[i], decoded_imgs[i]).numpy()
        return reconstruction_loss
    
    def _plot_results(self, n, test_noisy, test, decoded_imgs):
        """
        The _plot_results method is used to show the comparison of the noised images and the denoised ones, and the
        original images without noise (if they are used as argument).
        :param n: it is the number of images to compare
        :param test_noisy: it is the set of noised images
        :param test: it is the set of original images
        :param decoded_imgs: it is the set of denoised images
        """
        
        if test is None:
            subp = 2
        else:
            subp = 3
        
        if self.image_dimension[2] == 1:
            dimension = (self.image_dimension[0], self.image_dimension[1])
        else:
            dimension = (self.image_dimension[0], self.image_dimension[1], self.image_dimension[2])

        for i in range(n):

            # display noisy
            ax = plt.subplot(subp, n, i + 1)
            plt.imshow(test_noisy[i].reshape(dimension))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(subp, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(dimension))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if not (test is None):
                # display original
                ax = plt.subplot(subp, n, i + 1 + n + n)
                plt.imshow(test[i].reshape(dimension))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()
        
        
    def _preprocessing(self, data):
        """
        The _preprocessing method is used in order to reshape the images to a common size and to normalize them.
        :param data: it is the dataset of images
        :return: the preprocessed data
        """
        data = np.reshape(data, (len(data), self.image_dimension[0], self.image_dimension[1], self.image_dimension[2]))
        return data.astype('float32') / data.max()

    def set_name(self, name):
        """
        The set_name method is used to change the name of the encoder.
        :param name: it is the name of the encoder
        """
        self.name = name

    def export_weights(self):
        """
        The export_weights method is used to save the weights of the fitted CNN.
        """
        self.autoencoder.save_weights(self.name)

    def import_weights(self, name):
        """
        The import_weights method is used to load the weights of a previously fitted CNN.
        :param name: it is the name of the file (with its path) which contains the weights to import
        """
        self.autoencoder.load_weights(name)

    def set_input_dimensions(self, dim):
        """
        The set_input_dimensions method is used to set the weights of the CNN input layer.
        :param dim: it is the list of the two dimensions
        """
        if len(dim) == 2:
            self.image_dimension = (dim[0], dim[1], 1)
        else:
            self.image_dimension = dim
        self.autoencoder = self.autoencoder_creation()


if __name__ == "__main__":
    print("The image_denoiser class allows to create autoencoders which filter out noise from input images.")
    print("In particular, the autoencoder object is a CNN, which takes 28 x 28 images (otherwise, it automatically,"
          " resamples them)")
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
    print("\n     ID = image_denoiser()")
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
