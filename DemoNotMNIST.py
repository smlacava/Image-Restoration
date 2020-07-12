  
"""notMNIST DEMO"""
# This is a demonstration of the use of the image denoiser with the notMNIST dataset composed by images of character in different fonts.
# In paricular, a Gaussian noise matrix is applied to the image of each digit.from matplotlib import pyplot as plt
import numpy as np
import gzip
from image_denoiser import image_denoiser


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28,28)
        return data
    
x_train = extract_data('notMNIST-to-MNIST-master/train-images-idx3-ubyte.gz', 60000)
x_test = extract_data('notMNIST-to-MNIST-master/t10k-images-idx3-ubyte.gz', 10000)


# Now, it is necessary to add some noise to the images, in order to create the noised datasets, both for the testing set
# and the test set.
# In particular, the Gaussian noise distribution has zero-mean (loc) and unitary standard deviation (scale), and the
# resulting values are multiplied by 32.
noise_factor = 32
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# The images are clipped between 0 and 255 (the denoiser will automatically normalize them between 0 and 1).
x_train_noisy = np.clip(x_train_noisy, 0., 255.)
x_test_noisy = np.clip(x_test_noisy, 0., 255.)

# Here, it is possible to see an image of the dataset and the corresponding noised one.
ax = plt.subplot(1, 2, 1)
plt.imshow(x_test[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 2, 2)
plt.imshow(x_test_noisy[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()


"""Creation and fitting of the denoiser"""
# When the datasets are ready, it is possible to create the denoiser and to train it.
# Furthermore, it is possible to give it a name (and a path), in order to export the weights for further analysis and to
# avoid repeating the training process.
filename = 'Denoiser'    # This is the name (and the path) of the denoiser
ID = image_denoiser()    # Creation of the denoiser
ID.set_name(filename)    # Assignment of the chosen name

# Training of the denoiser, using as validation set a fraction equal to 0.2 (20%) of the training set, on 200 epochs and
# with the size of the batches of data equal to 128.
# Note that, even if the number of epochs is 200, the algorithm will automatically stop the training if the CNN does not
# improve (reduce) the value of the validation loss after 3 epochs, and however only the weights with the minimum
# validation loss value are saved at the end of the process.
ID.fit(x_train_noisy, x_train, 0.2, epochs=300, batch_size=128)

"""Test of the denoiser"""
# Finally, it is possible to predict (denoise) the images belonging to the test set and check if the results are
# acceptable, comparing the resulting denoised images with the original ones.
# In particular, the first 10 denoised images are visually compared with the corresponding noised and original ones.
# If the result is satisfactory, it is possible to export the resulting weights through the corresponding export_weights
# method (in order to import them, it is possible to use the import_weights method).
denoised = ID.predict(x_test_noisy, x_test, 10)

# It is possible show one of the results singularly, in order to have a better view
ax = plt.subplot(1, 2, 1)
plt.imshow(x_test_noisy[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 2, 2)
plt.imshow(denoised[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
ID.export_weights()
