from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


class image_denoiser():
    def __init__(self, name = 'Image_Denoiser'):
        self.name = name
        input_img = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder = autoencoder


    def fit(self, train_noisy, train, val_noisy = None, val = None, epochs = 100, batch_size = 128, shuffle = True):
        if val_noisy is None or val is None:
            self.autoencoder.fit(train_noisy, train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
        else:
            self.autoencoder.fit(x_train_noisy, x_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(x_test_noisy, x_test),
                            callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    def predict(self, test_noisy, test = None, n = None):
        decoded_imgs = self.autoencoder.predict(test)
        if not(n is None or test is None):
            self._plot_results(n, test_noisy, test, decoded_imgs)


    def _plot_results(self, n, test_noisy, test, decoded_imgs):
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noisy
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(test_noisy[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        
    def export_weights(self):
        self.autoencoder.save_weights(self.name)

    def import_weights(self, name):
        self.autoencoder.load_weights(name)

    def set_name(self, name):
        self.name = name


(x_train, _), (x_test, _) = mnist.load_data()



x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
ID = image_denoiser()
ID.fit(x_train_noisy, x_train, epochs = 10, batch_size = 128)
ID.predict(x_test_noisy, x_test, 10)
